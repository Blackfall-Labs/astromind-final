//! Trophic chemical identifiers and vocabulary registry.

use std::collections::HashMap;

/// A developmental trophic chemical identifier.
/// Separate from runtime neuromodulators (DA, ACh, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrophicChemicalId(pub String);

impl TrophicChemicalId {
    pub fn new(name: &str) -> Self {
        TrophicChemicalId(name.to_string())
    }
}

/// Registry of trophic chemicals in a program.
pub struct TrophicVocabulary {
    chemicals: Vec<TrophicChemicalId>,
    index: HashMap<String, usize>,
}

/// Built-in chemicals that don't need to be declared.
const BUILTIN_CHEMICALS: &[&str] = &[
    "DA", "GABA", "ACh", "BDNF", "Wnt", "BMP", "FGF", "EphrinA", "NGF",
];

impl TrophicVocabulary {
    pub fn new() -> Self {
        let mut vocab = TrophicVocabulary {
            chemicals: Vec::new(),
            index: HashMap::new(),
        };
        for name in BUILTIN_CHEMICALS {
            vocab.register(name);
        }
        vocab
    }

    pub fn register(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.index.get(name) {
            return idx;
        }
        let idx = self.chemicals.len();
        self.chemicals.push(TrophicChemicalId::new(name));
        self.index.insert(name.to_string(), idx);
        idx
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn is_builtin(name: &str) -> bool {
        BUILTIN_CHEMICALS.contains(&name)
    }

    pub fn len(&self) -> usize {
        self.chemicals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chemicals.is_empty()
    }
}

impl Default for TrophicVocabulary {
    fn default() -> Self {
        Self::new()
    }
}
