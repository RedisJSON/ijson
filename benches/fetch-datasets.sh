#!/usr/bin/env bash
# Download the benchmark datasets for `cargo bench --bench corpus` from their pinned mirror.
# Every asset is checked against benches/datasets.sha256 before it is unpacked; run with --help for details.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SUMS=$HERE/datasets.sha256
MIRROR=${IJSON_DATASETS_MIRROR:-https://s3.amazonaws.com/benchmarks.redislabs/ijson/datasets-v1}
DIR=$HERE/datasets

usage() {
  cat <<EOF
usage: benches/fetch-datasets.sh [--dir DIR] [quick|standard|all|FAMILY...]

Downloads JSON datasets into DIR (default: benches/datasets) from the public
mirror s3.amazonaws.com/benchmarks.redislabs/ijson/datasets-v1, verifies each
asset against benches/datasets.sha256, and unpacks it. Re-runs skip assets that
were already verified and unpacked. Needs curl, zstd and tar.

Presets (download -> unpacked):
  quick      simdjson extras floatdata                  165 MB -> 0.57 GB  (default)
  standard   quick + generated ncbi                     252 MB -> 3.9 GB
  all        standard + gharchive pison                 1.7 GB -> 25 GB

Families (download -> unpacked):
  simdjson   simdjson/simdjson-data: jsonexamples (twitter, citm_catalog,
             canada, gsoc-2018, ...), small, jsonchecker, minefield    5 MB -> 29 MB
  extras     other simdjson/Lemire repository data files              8 MB -> 112 MB
  floatdata  fastfloat/float-data number files + 768-float NDJSON   152 MB -> 425 MB
  generated  simdjson's generators (kostya, large_random, amazon,
             tweets, timeline, growing), fixed seed                   33 MB -> 2.1 GB
  ncbi       NCBI dbSNP RefSNP JSON, build 157 (public domain)        54 MB -> 1.2 GB
  gharchive  GH Archive hours, one event per line                    549 MB -> 8.3 GB
  pison      Pison (PVLDB 2021) evaluation data                      907 MB -> 13 GB

gharchive contains public GitHub user data (logins, commit author e-mails) and is
fetched only with 'all' or by naming it; do not commit it into a repository.
simdjson-data, float-data and Pison state no data licence; the mirror holds
unmodified copies for benchmarking, see each project for terms.

Environment:
  IJSON_DATASETS_MIRROR  base URL of another copy of the same assets (e.g. file:///srv/mirror);
                         the pinned sha256 list still applies.
EOF
}

die() { echo "fetch-datasets: $*" >&2; exit 1; }

FAMS=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --dir) [ $# -ge 2 ] || die "--dir needs a directory"; DIR=$2; shift ;;
    --dir=*) DIR=${1#--dir=} ;;
    quick) FAMS+=(simdjson extras floatdata) ;;
    standard) FAMS+=(simdjson extras floatdata generated ncbi) ;;
    all) FAMS+=(simdjson extras floatdata generated ncbi gharchive pison) ;;
    simdjson|simdjson-data) FAMS+=(simdjson) ;;
    extras|floatdata|generated|ncbi|gharchive|pison) FAMS+=("$1") ;;
    *) usage >&2; die "unknown preset or family: $1" ;;
  esac
  shift
done
[ ${#FAMS[@]} -gt 0 ] || FAMS=(simdjson extras floatdata)

command -v curl >/dev/null || die "curl not found"
command -v tar >/dev/null || die "tar not found"
if ! command -v zstd >/dev/null; then
  cat >&2 <<EOF
fetch-datasets: zstd not found; the assets are zstd-compressed. Install it with one of:
  Debian/Ubuntu:  sudo apt-get install zstd
  Fedora/RHEL:    sudo dnf install zstd
  macOS:          brew install zstd
EOF
  exit 1
fi
if command -v sha256sum >/dev/null; then sha() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else die "neither sha256sum nor shasum found"; fi
[ -f "$SUMS" ] || die "missing $SUMS"

mb() { awk -v b="$1" 'BEGIN { printf "%.1f MB", b / 1e6 }'; }

pattern() {  # family -> regex over asset names
  case $1 in
    simdjson) echo '^simdjson-data\.tar\.zst$' ;;
    generated) echo '^generated(\.tar\.zst|--.*)$' ;;
    gharchive|pison) echo "^$1--" ;;
    *) echo "^$1\\.tar\\.zst\$" ;;
  esac
}

mkdir -p "$DIR/.download" "$DIR/.verified"

# True when every file a verification mark lists (lines after the sha256) is still unpacked.
unpacked() {
  local f
  [ "$(wc -l < "$1")" -ge 2 ] || return 1  # a mark without a file list predates the check
  while IFS= read -r f; do
    [ -e "$DIR/$f" ] || return 1
  done < <(tail -n +2 "$1")
}
DIR=$(cd "$DIR" && pwd)
fetched=0 skipped=0 dl_bytes=0
seen=" "
for fam in "${FAMS[@]}"; do
  case $seen in *" $fam "*) continue ;; esac
  seen="$seen$fam "
  pat=$(pattern "$fam")
  while read -r sum asset; do
    [ -n "$asset" ] || continue
    echo "$asset" | grep -qE "$pat" || continue
    mark=$DIR/.verified/$asset
    if [ -f "$mark" ] && [ "$(head -n 1 "$mark")" = "$sum" ] && unpacked "$mark"; then
      skipped=$((skipped + 1)); echo "  ok       $asset (already verified)"; continue
    fi
    part=$DIR/.download/$asset
    curl -fsSL --retry 3 -o "$part" "$MIRROR/$asset" || die "download failed: $MIRROR/$asset"
    got=$(sha "$part")
    [ "$got" = "$sum" ] || { rm -f "$part"; die "sha256 mismatch for $asset: got $got, want $sum"; }
    size=$(wc -c < "$part" | tr -d ' ')
    case $asset in
      *.tar.zst)
        zstd -q -dc --long=27 "$part" | tar -xf - -C "$DIR"
        # The mark lists the unpacked files, so a re-run notices any that were deleted.
        { echo "$sum"; zstd -q -dc --long=27 "$part" | tar -tf - | grep -v '/$'; } > "$mark.tmp" ;;
      *--*.zst)
        f=${asset%.zst}; d=$DIR/${f%%--*}; out=$d/${f#*--}
        mkdir -p "$d"; zstd -q -dc --long=27 "$part" > "$out.tmp" && mv "$out.tmp" "$out"
        printf '%s\n%s\n' "$sum" "${f%%--*}/${f#*--}" > "$mark.tmp" ;;
      *) die "unknown asset type: $asset" ;;
    esac
    rm -f "$part"
    mv "$mark.tmp" "$mark"
    fetched=$((fetched + 1)); dl_bytes=$((dl_bytes + size))
    echo "  fetched  $asset ($(mb "$size"), sha256 ok)"
  done < "$SUMS"
done
rmdir "$DIR/.download" 2>/dev/null || true

fams=${seen# }; echo "fetch-datasets: ${fams% } -> $DIR"
echo "  $fetched asset(s) downloaded ($(mb "$dl_bytes")), $skipped already present; on disk: $(du -sh "$DIR" | cut -f1)"
echo "  next: IJSON_BENCH_DATASETS=$DIR cargo bench --no-default-features --bench corpus"
