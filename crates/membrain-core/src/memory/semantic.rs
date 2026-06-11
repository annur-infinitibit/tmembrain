//! Semantic memory types: facts, preferences, concepts, entities
//!
//! Semantic memories represent stable knowledge about the world.
//! They have slower decay rates and are typically distilled from
//! episodic memories or directly asserted.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::MemoryCommon;

/// Semantic memory containing knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    /// Common memory fields
    pub common: MemoryCommon,
    /// The specific semantic content
    pub content: SemanticContent,
}

impl SemanticMemory {
    /// Get text content for embedding/indexing
    pub fn text_content(&self) -> String {
        match &self.content {
            SemanticContent::Fact(f) => f.text_content(),
            SemanticContent::Preference(p) => p.text_content(),
            SemanticContent::Concept(c) => c.text_content(),
            SemanticContent::Entity(e) => e.text_content(),
        }
    }
}

/// Types of semantic content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "subtype", content = "data")]
pub enum SemanticContent {
    /// A factual statement
    Fact(FactMemory),
    /// A preference
    Preference(PreferenceMemory),
    /// A concept definition
    Concept(ConceptMemory),
    /// An entity (person, place, thing)
    Entity(EntityMemory),
}

/// A factual statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactMemory {
    /// The factual statement
    pub statement: String,
    /// Subject of the fact (optional structured extraction)
    pub subject: Option<String>,
    /// Predicate/relation
    pub predicate: Option<String>,
    /// Object of the fact
    pub object: Option<String>,
}

impl FactMemory {
    /// Create a new fact memory
    pub fn new(statement: impl Into<String>) -> Self {
        Self {
            statement: statement.into(),
            subject: None,
            predicate: None,
            object: None,
        }
    }

    /// Create with SPO triple
    pub fn with_triple(
        statement: impl Into<String>,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            statement: statement.into(),
            subject: Some(subject.into()),
            predicate: Some(predicate.into()),
            object: Some(object.into()),
        }
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        self.statement.clone()
    }
}

/// A preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceMemory {
    /// Who holds this preference
    pub holder: String,
    /// What the preference is about
    pub subject: String,
    /// The preferred value/option
    pub preference: String,
    /// Strength of preference
    pub strength: PreferenceStrength,
    /// Context in which this preference applies
    pub context: Option<String>,
    /// Alternative that was not preferred
    pub alternative: Option<String>,
    /// Reason for the preference (if known)
    pub reason: Option<String>,
}

impl PreferenceMemory {
    /// Create a new preference memory
    pub fn new(
        holder: impl Into<String>,
        subject: impl Into<String>,
        preference: impl Into<String>,
    ) -> Self {
        Self {
            holder: holder.into(),
            subject: subject.into(),
            preference: preference.into(),
            strength: PreferenceStrength::Moderate,
            context: None,
            alternative: None,
            reason: None,
        }
    }

    /// Set strength
    pub fn with_strength(mut self, strength: PreferenceStrength) -> Self {
        self.strength = strength;
        self
    }

    /// Set context
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set reason
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let strength_text = match self.strength {
            PreferenceStrength::Weak => "slightly prefers",
            PreferenceStrength::Moderate => "prefers",
            PreferenceStrength::Strong => "strongly prefers",
            PreferenceStrength::Absolute => "always wants",
        };

        let mut text = format!(
            "{} {} {} for {}",
            self.holder, strength_text, self.preference, self.subject
        );

        if let Some(ref context) = self.context {
            text.push_str(&format!(" (when {})", context));
        }

        if let Some(ref reason) = self.reason {
            text.push_str(&format!(" because {}", reason));
        }

        text
    }
}

/// Strength of a preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreferenceStrength {
    Weak,
    Moderate,
    Strong,
    Absolute,
}

impl PreferenceStrength {
    /// Convert to a numeric score (0.0-1.0)
    pub fn as_score(&self) -> f64 {
        match self {
            PreferenceStrength::Weak => 0.25,
            PreferenceStrength::Moderate => 0.5,
            PreferenceStrength::Strong => 0.75,
            PreferenceStrength::Absolute => 1.0,
        }
    }
}

/// A concept definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMemory {
    /// Name of the concept
    pub name: String,
    /// Definition/description
    pub definition: String,
    /// Examples of the concept
    pub examples: Vec<String>,
    /// Related concepts
    pub related: Vec<String>,
    /// Parent/broader concepts
    pub broader: Vec<String>,
    /// Child/narrower concepts
    pub narrower: Vec<String>,
    /// Domain or category
    pub domain: Option<String>,
}

impl ConceptMemory {
    /// Create a new concept memory
    pub fn new(name: impl Into<String>, definition: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            definition: definition.into(),
            examples: Vec::new(),
            related: Vec::new(),
            broader: Vec::new(),
            narrower: Vec::new(),
            domain: None,
        }
    }

    /// Add examples
    pub fn with_examples(mut self, examples: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.examples.extend(examples.into_iter().map(Into::into));
        self
    }

    /// Add related concepts
    pub fn with_related(mut self, related: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.related.extend(related.into_iter().map(Into::into));
        self
    }

    /// Set domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![
            format!("Concept: {}", self.name),
            format!("Definition: {}", self.definition),
        ];

        if !self.examples.is_empty() {
            parts.push(format!("Examples: {}", self.examples.join(", ")));
        }

        if !self.related.is_empty() {
            parts.push(format!("Related: {}", self.related.join(", ")));
        }

        if let Some(ref domain) = self.domain {
            parts.push(format!("Domain: {}", domain));
        }

        parts.join("\n")
    }
}

/// An entity (person, place, thing, organization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMemory {
    /// Name of the entity
    pub name: String,
    /// Type of entity
    pub entity_type: EntityType,
    /// Description
    pub description: Option<String>,
    /// Aliases/alternative names
    pub aliases: Vec<String>,
    /// Attributes (key-value pairs)
    pub attributes: HashMap<String, String>,
    /// Relationships to other entities
    pub relationships: Vec<EntityRelationship>,
    /// First seen/mentioned
    pub first_seen: DateTime<Utc>,
    /// Last seen/mentioned
    pub last_seen: DateTime<Utc>,
    /// Mention count
    pub mention_count: u32,
}

impl EntityMemory {
    /// Create a new entity memory
    pub fn new(name: impl Into<String>, entity_type: EntityType) -> Self {
        let now = Utc::now();
        Self {
            name: name.into(),
            entity_type,
            description: None,
            aliases: Vec::new(),
            attributes: HashMap::new(),
            relationships: Vec::new(),
            first_seen: now,
            last_seen: now,
            mention_count: 1,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add aliases
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases.extend(aliases.into_iter().map(Into::into));
        self
    }

    /// Add an attribute
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Add a relationship
    pub fn with_relationship(mut self, relationship: EntityRelationship) -> Self {
        self.relationships.push(relationship);
        self
    }

    /// Record a mention
    pub fn record_mention(&mut self) {
        self.last_seen = Utc::now();
        self.mention_count += 1;
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![format!("Entity: {} ({})", self.name, self.entity_type)];

        if let Some(ref desc) = self.description {
            parts.push(format!("Description: {}", desc));
        }

        if !self.aliases.is_empty() {
            parts.push(format!("Also known as: {}", self.aliases.join(", ")));
        }

        for (key, value) in &self.attributes {
            parts.push(format!("{}: {}", key, value));
        }

        for rel in &self.relationships {
            parts.push(format!(
                "Relationship: {} {} {}",
                self.name, rel.relation_type, rel.target_entity
            ));
        }

        parts.join("\n")
    }
}

/// Type of entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityType {
    Person,
    Organization,
    Place,
    Product,
    Event,
    Concept,
    Document,
    Technology,
    Other,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::Person => write!(f, "person"),
            EntityType::Organization => write!(f, "organization"),
            EntityType::Place => write!(f, "place"),
            EntityType::Product => write!(f, "product"),
            EntityType::Event => write!(f, "event"),
            EntityType::Concept => write!(f, "concept"),
            EntityType::Document => write!(f, "document"),
            EntityType::Technology => write!(f, "technology"),
            EntityType::Other => write!(f, "other"),
        }
    }
}

/// A relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
    /// The type of relationship
    pub relation_type: String,
    /// Target entity name
    pub target_entity: String,
    /// When this relationship was established
    pub established: Option<DateTime<Utc>>,
    /// Additional context
    pub context: Option<String>,
}

impl EntityRelationship {
    /// Create a new relationship
    pub fn new(relation_type: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            relation_type: relation_type.into(),
            target_entity: target.into(),
            established: None,
            context: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_memory_creation() {
        let fact = FactMemory::with_triple(
            "The Eiffel Tower is in Paris",
            "Eiffel Tower",
            "located in",
            "Paris",
        );

        assert_eq!(fact.subject, Some("Eiffel Tower".to_string()));
        assert!(fact.text_content().contains("Eiffel Tower"));
    }

    #[test]
    fn preference_memory_creation() {
        let pref = PreferenceMemory::new("user", "theme", "dark mode")
            .with_strength(PreferenceStrength::Strong)
            .with_reason("easier on the eyes");

        let text = pref.text_content();
        assert!(text.contains("strongly prefers"));
        assert!(text.contains("dark mode"));
        assert!(text.contains("easier on the eyes"));
    }

    #[test]
    fn concept_memory_creation() {
        let concept =
            ConceptMemory::new("Machine Learning", "A subset of AI that learns from data")
                .with_examples(vec!["neural networks", "decision trees"])
                .with_related(vec!["artificial intelligence", "deep learning"])
                .with_domain("computer science");

        let text = concept.text_content();
        assert!(text.contains("Machine Learning"));
        assert!(text.contains("neural networks"));
    }

    #[test]
    fn entity_memory_creation() {
        let entity = EntityMemory::new("Alice Smith", EntityType::Person)
            .with_description("Project manager")
            .with_aliases(vec!["Alice", "AS"])
            .with_attribute("role", "PM")
            .with_relationship(EntityRelationship::new("works at", "Acme Corp"));

        let text = entity.text_content();
        assert!(text.contains("Alice Smith"));
        assert!(text.contains("person"));
        assert!(text.contains("works at"));
    }

    #[test]
    fn preference_strength_scores() {
        assert!(PreferenceStrength::Weak.as_score() < PreferenceStrength::Moderate.as_score());
        assert!(PreferenceStrength::Moderate.as_score() < PreferenceStrength::Strong.as_score());
        assert!(PreferenceStrength::Strong.as_score() < PreferenceStrength::Absolute.as_score());
    }
}
