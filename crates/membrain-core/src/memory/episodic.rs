//! Episodic memory types: conversations, events, observations
//!
//! Episodic memories are high-detail records of specific experiences.
//! They typically have faster decay rates and are candidates for
//! consolidation into semantic memories.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::MemoryCommon;
use crate::types::SessionId;

/// Episodic memory containing experiential data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// Common memory fields
    pub common: MemoryCommon,
    /// The specific episodic content
    pub content: EpisodicContent,
}

impl EpisodicMemory {
    /// Get text content for embedding/indexing
    pub fn text_content(&self) -> String {
        match &self.content {
            EpisodicContent::Conversation(c) => c.text_content(),
            EpisodicContent::Event(e) => e.text_content(),
            EpisodicContent::Observation(o) => o.text_content(),
        }
    }
}

/// Types of episodic content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "subtype", content = "data")]
pub enum EpisodicContent {
    /// A conversation or part of one
    Conversation(ConversationMemory),
    /// A significant event
    Event(EventMemory),
    /// An observation about the world
    Observation(ObservationMemory),
}

/// A conversation or conversation segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMemory {
    /// The session this conversation belongs to
    pub session_id: SessionId,
    /// Messages in this conversation segment
    pub messages: Vec<Message>,
    /// Summary of the conversation
    pub summary: Option<String>,
    /// Topics discussed
    pub topics: Vec<String>,
    /// When the conversation started
    pub started_at: DateTime<Utc>,
    /// When the conversation ended (if known)
    pub ended_at: Option<DateTime<Utc>>,
    /// Sentiment/tone of the conversation
    pub sentiment: Option<String>,
    /// Whether this is a complete conversation or a segment
    pub is_complete: bool,
}

impl ConversationMemory {
    /// Create a new conversation memory
    pub fn new(session_id: SessionId, messages: Vec<Message>) -> Self {
        Self {
            session_id,
            messages,
            summary: None,
            topics: Vec::new(),
            started_at: Utc::now(),
            ended_at: None,
            sentiment: None,
            is_complete: false,
        }
    }

    /// Set the summary
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Add topics
    pub fn with_topics(mut self, topics: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.topics.extend(topics.into_iter().map(Into::into));
        self
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref summary) = self.summary {
            parts.push(format!("Summary: {}", summary));
        }

        if !self.topics.is_empty() {
            parts.push(format!("Topics: {}", self.topics.join(", ")));
        }

        for msg in &self.messages {
            parts.push(format!("{}: {}", msg.role, msg.content));
        }

        parts.join("\n")
    }
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the speaker
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
    /// When this message was sent
    pub timestamp: DateTime<Utc>,
    /// Optional name for the speaker
    pub name: Option<String>,
    /// Additional metadata (e.g., tool calls, function results)
    pub metadata: Option<serde_json::Value>,
}

impl Message {
    /// Create a new message
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp: Utc::now(),
            name: None,
            metadata: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    /// Set the speaker name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Role of a message sender
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
    Function,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::System => write!(f, "system"),
            MessageRole::Tool => write!(f, "tool"),
            MessageRole::Function => write!(f, "function"),
        }
    }
}

/// A significant event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMemory {
    /// Type of event
    pub event_type: String,
    /// Description of what happened
    pub description: String,
    /// When the event occurred
    pub occurred_at: DateTime<Utc>,
    /// Duration if applicable
    pub duration: Option<chrono::Duration>,
    /// Entities involved
    pub participants: Vec<String>,
    /// Location or context
    pub location: Option<String>,
    /// Outcome or result
    pub outcome: Option<String>,
    /// Importance/significance (0.0-1.0)
    pub significance: f64,
    /// Emotional valence (-1.0 to 1.0, negative to positive)
    pub emotional_valence: Option<f64>,
}

impl EventMemory {
    /// Create a new event memory
    pub fn new(event_type: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            description: description.into(),
            occurred_at: Utc::now(),
            duration: None,
            participants: Vec::new(),
            location: None,
            outcome: None,
            significance: 0.5,
            emotional_valence: None,
        }
    }

    /// Set significance
    pub fn with_significance(mut self, significance: f64) -> Self {
        self.significance = significance.clamp(0.0, 1.0);
        self
    }

    /// Add participants
    pub fn with_participants(
        mut self,
        participants: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.participants
            .extend(participants.into_iter().map(Into::into));
        self
    }

    /// Set location
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Set outcome
    pub fn with_outcome(mut self, outcome: impl Into<String>) -> Self {
        self.outcome = Some(outcome.into());
        self
    }

    /// Get text content for embedding.
    ///
    /// For conversation messages, the raw description is used directly to avoid
    /// polluting embeddings with a noisy `Event: conversation_message -` prefix.
    /// All other event types retain the prefix for context.
    pub fn text_content(&self) -> String {
        let first_line = if self.event_type == "conversation_message" {
            self.description.clone()
        } else {
            format!("Event: {} - {}", self.event_type, self.description)
        };

        let mut parts = vec![first_line];

        if !self.participants.is_empty() {
            parts.push(format!("Participants: {}", self.participants.join(", ")));
        }

        if let Some(ref location) = self.location {
            parts.push(format!("Location: {}", location));
        }

        if let Some(ref outcome) = self.outcome {
            parts.push(format!("Outcome: {}", outcome));
        }

        parts.join("\n")
    }
}

/// An observation about the world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationMemory {
    /// What was observed
    pub content: String,
    /// Category of observation
    pub category: Option<String>,
    /// Subject of observation
    pub subject: Option<String>,
    /// When this was observed
    pub observed_at: DateTime<Utc>,
    /// Context in which it was observed
    pub context: Option<String>,
    /// Whether this has been verified
    pub verified: bool,
    /// Source of the observation
    pub source: Option<String>,
}

impl ObservationMemory {
    /// Create a new observation
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            category: None,
            subject: None,
            observed_at: Utc::now(),
            context: None,
            verified: false,
            source: None,
        }
    }

    /// Set the category
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set the subject
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    /// Mark as verified
    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![format!("Observation: {}", self.content)];

        if let Some(ref category) = self.category {
            parts.push(format!("Category: {}", category));
        }

        if let Some(ref subject) = self.subject {
            parts.push(format!("Subject: {}", subject));
        }

        if let Some(ref context) = self.context {
            parts.push(format!("Context: {}", context));
        }

        parts.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_memory_creation() {
        let session = SessionId::new();
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi! How can I help?"),
        ];

        let conv = ConversationMemory::new(session, messages)
            .with_summary("A greeting exchange")
            .with_topics(vec!["greeting", "introduction"]);

        assert_eq!(conv.messages.len(), 2);
        assert_eq!(conv.topics.len(), 2);
        assert!(conv.summary.is_some());
    }

    #[test]
    fn event_memory_creation() {
        let event = EventMemory::new("meeting", "Team standup meeting")
            .with_significance(0.7)
            .with_participants(vec!["Alice", "Bob"])
            .with_location("Conference Room A")
            .with_outcome("Discussed sprint goals");

        assert_eq!(event.significance, 0.7);
        assert_eq!(event.participants.len(), 2);
        assert!(event.location.is_some());
    }

    #[test]
    fn observation_memory_creation() {
        let obs = ObservationMemory::new("User prefers dark mode")
            .with_category("preference")
            .with_subject("user interface")
            .verified();

        assert!(obs.verified);
        assert_eq!(obs.category, Some("preference".to_string()));
    }

    #[test]
    fn event_text_content_conversation_message_is_clean() {
        let event = EventMemory::new(
            "conversation_message",
            "[Speaker: Angela] 8 May | I love pizza",
        );
        let text = event.text_content();
        // Should NOT contain the "Event: conversation_message -" prefix
        assert!(!text.contains("Event:"));
        assert!(text.starts_with("[Speaker: Angela] 8 May | I love pizza"));
    }

    #[test]
    fn event_text_content_other_type_has_prefix() {
        let event = EventMemory::new("meeting", "Team standup meeting");
        let text = event.text_content();
        assert!(text.starts_with("Event: meeting - Team standup meeting"));
    }

    #[test]
    fn text_content_generation() {
        let session = SessionId::new();
        let messages = vec![
            Message::user("What's the weather?"),
            Message::assistant("It's sunny today!"),
        ];

        let conv = ConversationMemory::new(session, messages).with_summary("Weather inquiry");

        let text = conv.text_content();
        assert!(text.contains("Weather inquiry"));
        assert!(text.contains("user: What's the weather?"));
        assert!(text.contains("assistant: It's sunny today!"));
    }
}
