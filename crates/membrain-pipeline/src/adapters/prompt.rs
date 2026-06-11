//! Prompt adapter for injecting memories into LLM prompts

use serde::{Deserialize, Serialize};

use crate::retrieval::RetrievedMemory;
use membrain_core::memory::MemoryCategory;

/// Format for memory context in prompts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptFormat {
    /// Simple list format
    List,
    /// XML-style tags
    Xml,
    /// Markdown format
    Markdown,
    /// JSON format
    Json,
}

/// Memory context formatted for a prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    /// Formatted context string
    pub content: String,
    /// Number of memories included
    pub memory_count: usize,
    /// Estimated token count
    pub token_count: usize,
    /// Categories represented
    pub categories: Vec<MemoryCategory>,
}

/// Adapter for formatting memories into prompts
pub struct PromptAdapter {
    format: PromptFormat,
    max_tokens: usize,
    include_metadata: bool,
    category_headers: bool,
}

impl PromptAdapter {
    /// Create a new prompt adapter
    pub fn new(format: PromptFormat) -> Self {
        Self {
            format,
            max_tokens: 4000,
            include_metadata: false,
            category_headers: true,
        }
    }

    /// Set maximum tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Include metadata in output
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Include category headers
    pub fn with_category_headers(mut self, include: bool) -> Self {
        self.category_headers = include;
        self
    }

    /// Format memories into a prompt context
    pub fn format_context(&self, memories: &[RetrievedMemory]) -> MemoryContext {
        if memories.is_empty() {
            return MemoryContext {
                content: String::new(),
                memory_count: 0,
                token_count: 0,
                categories: vec![],
            };
        }

        // Group by category if using headers
        let grouped = if self.category_headers {
            self.group_by_category(memories)
        } else {
            vec![(None, memories.to_vec())]
        };

        let content = match self.format {
            PromptFormat::List => self.format_list(&grouped),
            PromptFormat::Xml => self.format_xml(&grouped),
            PromptFormat::Markdown => self.format_markdown(&grouped),
            PromptFormat::Json => self.format_json(&grouped),
        };

        let categories: Vec<MemoryCategory> = memories
            .iter()
            .map(|m| m.memory.memory_type().category())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let token_count = content.split_whitespace().count() * 4 / 3;

        MemoryContext {
            content,
            memory_count: memories.len(),
            token_count,
            categories,
        }
    }

    fn group_by_category(
        &self,
        memories: &[RetrievedMemory],
    ) -> Vec<(Option<MemoryCategory>, Vec<RetrievedMemory>)> {
        use std::collections::HashMap;

        let mut groups: HashMap<MemoryCategory, Vec<RetrievedMemory>> = HashMap::new();

        for memory in memories {
            let category = memory.memory.memory_type().category();
            groups.entry(category).or_default().push(memory.clone());
        }

        let mut result: Vec<_> = groups
            .into_iter()
            .map(|(cat, mems)| (Some(cat), mems))
            .collect();

        // Sort by category for consistent output
        result.sort_by_key(|(cat, _)| match cat {
            Some(MemoryCategory::Semantic) => 0,
            Some(MemoryCategory::Episodic) => 1,
            Some(MemoryCategory::Procedural) => 2,
            Some(MemoryCategory::AgentState) => 3,
            None => 4,
        });

        result
    }

    fn format_list(&self, grouped: &[(Option<MemoryCategory>, Vec<RetrievedMemory>)]) -> String {
        let mut lines = Vec::new();

        for (category, memories) in grouped {
            if let Some(cat) = category {
                lines.push(format!("\n{}:", self.category_label(cat)));
            }

            for memory in memories {
                let line = if self.include_metadata {
                    format!(
                        "- {} (confidence: {:.0}%)",
                        memory.text_content,
                        memory.memory.confidence().value() * 100.0
                    )
                } else {
                    format!("- {}", memory.text_content)
                };
                lines.push(line);
            }
        }

        lines.join("\n")
    }

    fn format_xml(&self, grouped: &[(Option<MemoryCategory>, Vec<RetrievedMemory>)]) -> String {
        let mut xml = String::from("<memory_context>\n");

        for (category, memories) in grouped {
            if let Some(cat) = category {
                xml.push_str(&format!("  <{}>", self.category_tag(cat)));
            }

            for memory in memories {
                if self.include_metadata {
                    xml.push_str(&format!(
                        "\n    <memory confidence=\"{:.2}\" type=\"{}\">\n      {}\n    </memory>",
                        memory.memory.confidence().value(),
                        memory.memory.memory_type(),
                        memory.text_content
                    ));
                } else {
                    xml.push_str(&format!("\n    <memory>{}</memory>", memory.text_content));
                }
            }

            if let Some(cat) = category.as_ref() {
                xml.push_str(&format!("\n  </{}>\n", self.category_tag(cat)));
            }
        }

        xml.push_str("</memory_context>");
        xml
    }

    fn format_markdown(
        &self,
        grouped: &[(Option<MemoryCategory>, Vec<RetrievedMemory>)],
    ) -> String {
        let mut md = String::from("## Relevant Context\n\n");

        for (category, memories) in grouped {
            if let Some(cat) = category {
                md.push_str(&format!("### {}\n\n", self.category_label(cat)));
            }

            for memory in memories {
                if self.include_metadata {
                    md.push_str(&format!(
                        "- **{}** _(confidence: {:.0}%)_\n",
                        memory.text_content,
                        memory.memory.confidence().value() * 100.0
                    ));
                } else {
                    md.push_str(&format!("- {}\n", memory.text_content));
                }
            }

            md.push('\n');
        }

        md
    }

    fn format_json(&self, grouped: &[(Option<MemoryCategory>, Vec<RetrievedMemory>)]) -> String {
        #[derive(Serialize)]
        struct JsonMemory {
            content: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            confidence: Option<f64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            memory_type: Option<String>,
        }

        #[derive(Serialize)]
        struct JsonContext {
            memories: Vec<JsonMemory>,
        }

        let memories: Vec<JsonMemory> = grouped
            .iter()
            .flat_map(|(_, mems)| mems)
            .map(|m| JsonMemory {
                content: m.text_content.clone(),
                confidence: if self.include_metadata {
                    Some(m.memory.confidence().value())
                } else {
                    None
                },
                memory_type: if self.include_metadata {
                    Some(m.memory.memory_type().to_string())
                } else {
                    None
                },
            })
            .collect();

        let context = JsonContext { memories };
        serde_json::to_string_pretty(&context).unwrap_or_default()
    }

    fn category_label(&self, category: &MemoryCategory) -> &'static str {
        match category {
            MemoryCategory::Semantic => "Known Facts",
            MemoryCategory::Episodic => "Past Interactions",
            MemoryCategory::Procedural => "Procedures",
            MemoryCategory::AgentState => "Current State",
        }
    }

    fn category_tag(&self, category: &MemoryCategory) -> &'static str {
        match category {
            MemoryCategory::Semantic => "semantic",
            MemoryCategory::Episodic => "episodic",
            MemoryCategory::Procedural => "procedural",
            MemoryCategory::AgentState => "agent_state",
        }
    }
}

impl Default for PromptAdapter {
    fn default() -> Self {
        Self::new(PromptFormat::Markdown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use membrain_core::memory::{FactMemory, MemoryCommon, SemanticContent, SemanticMemory};
    use membrain_core::types::{AgentId, Confidence, Provenance, Source};

    fn create_retrieved(text: &str) -> RetrievedMemory {
        let agent_id = AgentId::new();
        let prov = Provenance::new_direct(Source::user_input("test"), agent_id);
        let common = MemoryCommon::new(agent_id, prov).with_confidence(Confidence::new(0.85));

        let memory = membrain_core::memory::Memory::Semantic(SemanticMemory {
            common,
            content: SemanticContent::Fact(FactMemory::new(text)),
        });

        RetrievedMemory {
            id: *memory.id(),
            memory,
            score: 0.9,
            estimated_tokens: text.split_whitespace().count(),
            text_content: text.to_string(),
        }
    }

    #[test]
    fn test_list_format() {
        let adapter = PromptAdapter::new(PromptFormat::List);

        let memories = vec![
            create_retrieved("User prefers dark mode"),
            create_retrieved("User likes Rust programming"),
        ];

        let context = adapter.format_context(&memories);

        assert!(context.content.contains("User prefers dark mode"));
        assert!(context.content.contains("User likes Rust"));
        assert_eq!(context.memory_count, 2);
    }

    #[test]
    fn test_xml_format() {
        let adapter = PromptAdapter::new(PromptFormat::Xml);

        let memories = vec![create_retrieved("Test fact")];

        let context = adapter.format_context(&memories);

        assert!(context.content.contains("<memory_context>"));
        assert!(context.content.contains("Test fact"));
        assert!(context.content.contains("</memory_context>"));
    }

    #[test]
    fn test_markdown_format() {
        let adapter = PromptAdapter::new(PromptFormat::Markdown);

        let memories = vec![create_retrieved("Test fact")];

        let context = adapter.format_context(&memories);

        assert!(context.content.contains("## Relevant Context"));
        assert!(context.content.contains("- Test fact"));
    }

    #[test]
    fn test_json_format() {
        let adapter = PromptAdapter::new(PromptFormat::Json);

        let memories = vec![create_retrieved("Test fact")];

        let context = adapter.format_context(&memories);

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&context.content).unwrap();
        assert!(parsed["memories"].is_array());
    }

    #[test]
    fn test_with_metadata() {
        let adapter = PromptAdapter::new(PromptFormat::List).with_metadata(true);

        let memories = vec![create_retrieved("Test fact")];

        let context = adapter.format_context(&memories);

        assert!(context.content.contains("confidence"));
    }

    #[test]
    fn test_empty_memories() {
        let adapter = PromptAdapter::new(PromptFormat::Markdown);

        let context = adapter.format_context(&[]);

        assert!(context.content.is_empty());
        assert_eq!(context.memory_count, 0);
    }
}
