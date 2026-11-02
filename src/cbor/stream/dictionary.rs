use std::num::NonZeroUsize;

use hashbrown::HashMap;

use super::{next_buckets, table_alloc_bytes, table_bytes, AddressTable, Slot};
use crate::IString;

/// Text supplied to a dictionary, with an optional interned allocation identity.
#[derive(Debug, Clone, Copy)]
pub struct StreamString<'a> {
    text: &'a str,
    identity: Option<(NonZeroUsize, usize)>,
}

impl<'a> StreamString<'a> {
    /// The string's UTF-8 contents.
    pub fn as_str(&self) -> &'a str {
        self.text
    }

    /// An interned allocation's address and owner count, when available.
    pub fn identity(&self) -> Option<(NonZeroUsize, usize)> {
        self.identity
    }
}

impl<'a> From<&'a str> for StreamString<'a> {
    fn from(text: &'a str) -> Self {
        Self {
            text,
            identity: None,
        }
    }
}

impl<'a> From<&'a String> for StreamString<'a> {
    fn from(text: &'a String) -> Self {
        text.as_str().into()
    }
}

impl<'a> From<&'a IString> for StreamString<'a> {
    fn from(text: &'a IString) -> Self {
        Self {
            text: text.as_str(),
            identity: text
                .heap_identity()
                .map(|(address, _, owners)| (address, owners)),
        }
    }
}

/// String identity and storage used by a stream encoder.
///
/// Entries must retain their identity until `clear`. `insert` must account for all
/// entry allocations, including simultaneous old/new tables during growth.
/// IDs are assigned by the encoder; dictionaries must return them unchanged.
pub trait StringDictionary {
    /// Number of definitions in the current dictionary.
    fn len(&self) -> usize;

    /// Whether this dictionary has no definitions.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Releases definitions; implementations may retain their table allocation.
    fn clear(&mut self);

    /// Returns a known string's ID and its length when registered.
    fn get(&self, string: &StreamString<'_>) -> Option<(u32, usize)>;

    /// Whether a previously unseen string is eligible for a definition.
    fn can_define(&self, string: &StreamString<'_>) -> bool;

    /// Inserts an absent string if its peak allocation fits `budget`.
    /// A refusal must leave the dictionary unchanged.
    fn insert(&mut self, string: &StreamString<'_>, id: u32, budget: usize) -> bool;
}

impl<D: StringDictionary + ?Sized> StringDictionary for Box<D> {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn clear(&mut self) {
        (**self).clear();
    }
    fn get(&self, string: &StreamString<'_>) -> Option<(u32, usize)> {
        (**self).get(string)
    }
    fn can_define(&self, string: &StreamString<'_>) -> bool {
        (**self).can_define(string)
    }
    fn insert(&mut self, string: &StreamString<'_>, id: u32, budget: usize) -> bool {
        (**self).insert(string, id, budget)
    }
}

/// Recognizes interned strings without copying their bytes or retaining handles.
///
/// The caller must keep every defined string allocation alive and at the same
/// address until the stream ends. The encoder stores addresses, not owned handles;
/// dropping or moving those strings can make later references resolve to wrong text.
#[derive(Debug, Default)]
pub struct AddressDictionary {
    pub(super) entries: AddressTable,
}

impl AddressDictionary {
    /// Heap bytes held by the ID table.
    pub(super) fn used_bytes(&self) -> usize {
        table_bytes(self.entries.capacity())
    }

    /// Whether one more definition fits the budget, counting the old and the new
    /// allocation while the table grows. No allocation can fail gracefully, so the
    /// budget is checked in advance. A table that is full and cannot grow stays so,
    /// because nothing is added to it from then on.
    /// An explicit reset reuses its existing allocation for a new dictionary.
    fn has_room(&self, budget: usize) -> bool {
        let table_growth = if self.entries.len() == self.entries.capacity() {
            table_alloc_bytes(next_buckets(self.entries.capacity()))
        } else {
            0
        };
        (table_growth == 0 || self.used_bytes() + table_growth <= budget)
            && self.entries.len() < u32::MAX as usize
    }
}

impl StringDictionary for AddressDictionary {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn get(&self, string: &StreamString<'_>) -> Option<(u32, usize)> {
        let (address, _) = string.identity()?;
        self.entries
            .get(&address.get())
            .map(|slot| (slot.id, slot.len as usize))
    }

    fn can_define(&self, string: &StreamString<'_>) -> bool {
        // Occurrences left inline because the string had a single owner.
        string.identity().is_some_and(|(_, owners)| owners >= 2)
    }

    fn insert(&mut self, string: &StreamString<'_>, id: u32, budget: usize) -> bool {
        let Some((address, _)) = string.identity() else {
            return false;
        };
        let Ok(len) = u32::try_from(string.as_str().len()) else {
            return false;
        };
        if !self.has_room(budget) {
            return false;
        }
        self.entries.insert(address.get(), Slot { id, len });
        true
    }
}

/// Matches strings by contents and owns a copy of every definition.
///
/// Source strings may be dropped immediately after encoding a document. The
/// budget includes these copies as well as hash-table storage and growth.
#[derive(Debug, Default)]
pub struct ContentDictionary {
    entries: HashMap<Box<str>, u32>,
    string_bytes: usize,
}

impl ContentDictionary {
    fn allocation(buckets: usize) -> usize {
        buckets * (std::mem::size_of::<(Box<str>, u32)>() + 1) + 16
    }
}

impl StringDictionary for ContentDictionary {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.string_bytes = 0;
    }

    fn get(&self, string: &StreamString<'_>) -> Option<(u32, usize)> {
        let string = string.as_str();
        self.entries.get(string).map(|&id| (id, string.len()))
    }

    fn can_define(&self, _string: &StreamString<'_>) -> bool {
        true
    }

    fn insert(&mut self, string: &StreamString<'_>, id: u32, budget: usize) -> bool {
        let string = string.as_str();
        let capacity = self.entries.capacity();
        let current = if capacity == 0 {
            0
        } else {
            Self::allocation(super::buckets_for_capacity(capacity))
        };
        let growth = if self.entries.len() == capacity {
            Self::allocation(next_buckets(capacity))
        } else {
            0
        };
        let required = current
            .saturating_add(growth)
            .saturating_add(self.string_bytes)
            .saturating_add(string.len());
        if required > budget {
            return false;
        }
        self.entries.insert(Box::from(string), id);
        self.string_bytes += string.len();
        true
    }
}
