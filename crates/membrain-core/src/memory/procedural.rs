//! Procedural memory types: workflows, skills, patterns
//!
//! Procedural memories represent knowledge about how to do things.
//! They include workflows, learned skills, and recognized patterns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::MemoryCommon;

/// Procedural memory containing action knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralMemory {
    /// Common memory fields
    pub common: MemoryCommon,
    /// The specific procedural content
    pub content: ProceduralContent,
}

impl ProceduralMemory {
    /// Get text content for embedding/indexing
    pub fn text_content(&self) -> String {
        match &self.content {
            ProceduralContent::Workflow(w) => w.text_content(),
            ProceduralContent::Skill(s) => s.text_content(),
            ProceduralContent::Pattern(p) => p.text_content(),
            ProceduralContent::Case(c) => c.text_content(),
        }
    }
}

/// Types of procedural content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "subtype", content = "data")]
pub enum ProceduralContent {
    /// A multi-step workflow
    Workflow(WorkflowMemory),
    /// A learned skill
    Skill(SkillMemory),
    /// A recognized pattern
    Pattern(PatternMemory),
    /// A stored experience case for case-based reasoning
    Case(CaseMemory),
}

/// A multi-step workflow or process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowMemory {
    /// Name of the workflow
    pub name: String,
    /// Description of what this workflow accomplishes
    pub description: String,
    /// The steps in this workflow
    pub steps: Vec<StepDefinition>,
    /// Trigger conditions for this workflow
    pub triggers: Vec<String>,
    /// Required inputs
    pub inputs: Vec<WorkflowIO>,
    /// Expected outputs
    pub outputs: Vec<WorkflowIO>,
    /// Preconditions that must be met
    pub preconditions: Vec<String>,
    /// Postconditions after execution
    pub postconditions: Vec<String>,
    /// Success rate from past executions (0.0-1.0)
    pub success_rate: Option<f64>,
    /// Number of times this workflow has been executed
    pub execution_count: u32,
    /// Average execution time
    pub avg_duration: Option<chrono::Duration>,
}

impl WorkflowMemory {
    /// Create a new workflow memory
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            steps: Vec::new(),
            triggers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            success_rate: None,
            execution_count: 0,
            avg_duration: None,
        }
    }

    /// Add a step
    pub fn with_step(mut self, step: StepDefinition) -> Self {
        self.steps.push(step);
        self
    }

    /// Add steps
    pub fn with_steps(mut self, steps: impl IntoIterator<Item = StepDefinition>) -> Self {
        self.steps.extend(steps);
        self
    }

    /// Add triggers
    pub fn with_triggers(mut self, triggers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.triggers.extend(triggers.into_iter().map(Into::into));
        self
    }

    /// Add inputs
    pub fn with_inputs(mut self, inputs: impl IntoIterator<Item = WorkflowIO>) -> Self {
        self.inputs.extend(inputs);
        self
    }

    /// Add outputs
    pub fn with_outputs(mut self, outputs: impl IntoIterator<Item = WorkflowIO>) -> Self {
        self.outputs.extend(outputs);
        self
    }

    /// Record an execution
    pub fn record_execution(&mut self, success: bool, duration: Option<chrono::Duration>) {
        self.execution_count += 1;

        // Update success rate
        let prev_rate = self.success_rate.unwrap_or(0.0);
        let new_success = if success { 1.0 } else { 0.0 };
        self.success_rate = Some(
            (prev_rate * (self.execution_count - 1) as f64 + new_success)
                / self.execution_count as f64,
        );

        // Update average duration
        if let Some(dur) = duration {
            if let Some(prev_avg) = self.avg_duration {
                let total_secs =
                    prev_avg.num_seconds() * (self.execution_count - 1) as i64 + dur.num_seconds();
                self.avg_duration = Some(chrono::Duration::seconds(
                    total_secs / self.execution_count as i64,
                ));
            } else {
                self.avg_duration = Some(dur);
            }
        }
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![
            format!("Workflow: {}", self.name),
            format!("Description: {}", self.description),
        ];

        if !self.triggers.is_empty() {
            parts.push(format!("Triggers: {}", self.triggers.join(", ")));
        }

        if !self.steps.is_empty() {
            parts.push("Steps:".to_string());
            for (i, step) in self.steps.iter().enumerate() {
                parts.push(format!("  {}. {} - {}", i + 1, step.name, step.description));
            }
        }

        if !self.inputs.is_empty() {
            let inputs: Vec<_> = self.inputs.iter().map(|i| i.name.as_str()).collect();
            parts.push(format!("Inputs: {}", inputs.join(", ")));
        }

        if !self.outputs.is_empty() {
            let outputs: Vec<_> = self.outputs.iter().map(|o| o.name.as_str()).collect();
            parts.push(format!("Outputs: {}", outputs.join(", ")));
        }

        parts.join("\n")
    }
}

/// A step in a workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDefinition {
    /// Name of the step
    pub name: String,
    /// Description of what this step does
    pub description: String,
    /// Action to take (could be tool name, function, etc.)
    pub action: String,
    /// Parameters for the action
    pub parameters: HashMap<String, serde_json::Value>,
    /// Condition for executing this step (if any)
    pub condition: Option<String>,
    /// Step to go to on success (default: next)
    pub on_success: Option<String>,
    /// Step to go to on failure
    pub on_failure: Option<String>,
    /// Whether this step is optional
    pub optional: bool,
    /// Retry configuration
    pub retry_count: u32,
}

impl StepDefinition {
    /// Create a new step
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        action: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            action: action.into(),
            parameters: HashMap::new(),
            condition: None,
            on_success: None,
            on_failure: None,
            optional: false,
            retry_count: 0,
        }
    }

    /// Add a parameter
    pub fn with_parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Set condition
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }

    /// Mark as optional
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Set retry count
    pub fn with_retries(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }
}

/// Input or output definition for a workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowIO {
    /// Name of the input/output
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Type hint
    pub type_hint: Option<String>,
    /// Whether this is required
    pub required: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
}

impl WorkflowIO {
    /// Create a new required input/output
    pub fn required(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            type_hint: None,
            required: true,
            default: None,
        }
    }

    /// Create a new optional input/output
    pub fn optional(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            type_hint: None,
            required: false,
            default: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set type hint
    pub fn with_type(mut self, type_hint: impl Into<String>) -> Self {
        self.type_hint = Some(type_hint.into());
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }
}

/// A learned skill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMemory {
    /// Name of the skill
    pub name: String,
    /// Description of the skill
    pub description: String,
    /// Category of skill
    pub category: Option<String>,
    /// How this skill is typically invoked
    pub invocation: Option<String>,
    /// Examples of using this skill
    pub examples: Vec<SkillExample>,
    /// Prerequisites for using this skill
    pub prerequisites: Vec<String>,
    /// Proficiency level (0.0-1.0)
    pub proficiency: f64,
    /// Number of times used
    pub usage_count: u32,
    /// Last used
    pub last_used: DateTime<Utc>,
    /// Related skills
    pub related_skills: Vec<String>,
}

impl SkillMemory {
    /// Create a new skill memory
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            category: None,
            invocation: None,
            examples: Vec::new(),
            prerequisites: Vec::new(),
            proficiency: 0.5,
            usage_count: 0,
            last_used: Utc::now(),
            related_skills: Vec::new(),
        }
    }

    /// Set category
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set invocation
    pub fn with_invocation(mut self, invocation: impl Into<String>) -> Self {
        self.invocation = Some(invocation.into());
        self
    }

    /// Add an example
    pub fn with_example(mut self, example: SkillExample) -> Self {
        self.examples.push(example);
        self
    }

    /// Set proficiency
    pub fn with_proficiency(mut self, proficiency: f64) -> Self {
        self.proficiency = proficiency.clamp(0.0, 1.0);
        self
    }

    /// Record usage
    pub fn record_usage(&mut self, success: bool) {
        self.usage_count += 1;
        self.last_used = Utc::now();

        // Adjust proficiency based on success
        if success {
            self.proficiency = (self.proficiency + 0.1 * (1.0 - self.proficiency)).min(1.0);
        } else {
            self.proficiency = (self.proficiency - 0.05).max(0.0);
        }
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![
            format!("Skill: {}", self.name),
            format!("Description: {}", self.description),
        ];

        if let Some(ref category) = self.category {
            parts.push(format!("Category: {}", category));
        }

        if let Some(ref invocation) = self.invocation {
            parts.push(format!("Invocation: {}", invocation));
        }

        if !self.examples.is_empty() {
            parts.push("Examples:".to_string());
            for example in &self.examples {
                parts.push(format!("  - Input: {}", example.input));
                parts.push(format!("    Output: {}", example.output));
            }
        }

        if !self.prerequisites.is_empty() {
            parts.push(format!("Prerequisites: {}", self.prerequisites.join(", ")));
        }

        parts.join("\n")
    }
}

/// An example of skill usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillExample {
    /// Input that triggered the skill
    pub input: String,
    /// Output produced
    pub output: String,
    /// Context
    pub context: Option<String>,
    /// Whether this was successful
    pub success: bool,
}

impl SkillExample {
    /// Create a new example
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            context: None,
            success: true,
        }
    }
}

/// A recognized pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMemory {
    /// Name of the pattern
    pub name: String,
    /// Description of the pattern
    pub description: String,
    /// Pattern type/category
    pub pattern_type: PatternType,
    /// Conditions that indicate this pattern
    pub indicators: Vec<String>,
    /// What typically follows this pattern
    pub typical_response: Option<String>,
    /// Historical accuracy of pattern recognition
    pub accuracy: f64,
    /// Number of times this pattern was recognized
    pub recognition_count: u32,
    /// Examples where this pattern was observed
    pub observations: Vec<PatternObservation>,
}

impl PatternMemory {
    /// Create a new pattern memory
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        pattern_type: PatternType,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            pattern_type,
            indicators: Vec::new(),
            typical_response: None,
            accuracy: 0.5,
            recognition_count: 0,
            observations: Vec::new(),
        }
    }

    /// Add indicators
    pub fn with_indicators(
        mut self,
        indicators: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.indicators
            .extend(indicators.into_iter().map(Into::into));
        self
    }

    /// Set typical response
    pub fn with_response(mut self, response: impl Into<String>) -> Self {
        self.typical_response = Some(response.into());
        self
    }

    /// Record a pattern recognition
    pub fn record_recognition(&mut self, correct: bool) {
        self.recognition_count += 1;
        if correct {
            self.accuracy = (self.accuracy * (self.recognition_count - 1) as f64 + 1.0)
                / self.recognition_count as f64;
        } else {
            self.accuracy = (self.accuracy * (self.recognition_count - 1) as f64)
                / self.recognition_count as f64;
        }
    }

    /// Add an observation
    pub fn add_observation(&mut self, observation: PatternObservation) {
        self.observations.push(observation);
        // Keep only recent observations
        if self.observations.len() > 100 {
            self.observations.remove(0);
        }
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![
            format!("Pattern: {} ({})", self.name, self.pattern_type),
            format!("Description: {}", self.description),
        ];

        if !self.indicators.is_empty() {
            parts.push(format!("Indicators: {}", self.indicators.join(", ")));
        }

        if let Some(ref response) = self.typical_response {
            parts.push(format!("Typical response: {}", response));
        }

        parts.join("\n")
    }
}

/// Type of pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternType {
    /// User behavior pattern
    UserBehavior,
    /// Conversation pattern
    Conversation,
    /// Error/failure pattern
    Error,
    /// Success pattern
    Success,
    /// Temporal pattern (time-based)
    Temporal,
    /// Contextual pattern
    Contextual,
    /// Other
    Other,
}

impl std::fmt::Display for PatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternType::UserBehavior => write!(f, "user_behavior"),
            PatternType::Conversation => write!(f, "conversation"),
            PatternType::Error => write!(f, "error"),
            PatternType::Success => write!(f, "success"),
            PatternType::Temporal => write!(f, "temporal"),
            PatternType::Contextual => write!(f, "contextual"),
            PatternType::Other => write!(f, "other"),
        }
    }
}

/// A stored experience case for case-based reasoning.
///
/// Cases represent past execution experiences: the problem faced, the plan
/// chosen, the outcome observed, and a reward signal indicating success.
/// They are used as few-shot examples to guide future planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseMemory {
    /// The problem or query that was addressed
    pub problem: String,
    /// The plan that was executed
    pub plan: String,
    /// The observed outcome
    pub outcome: String,
    /// Reward signal (1.0 = full success, 0.0 = failure)
    pub reward: f64,
    /// Optional domain tag for scoping retrieval
    pub domain: Option<String>,
    /// Number of times this case has been retrieved
    pub retrieval_count: u32,
    /// When this case was last retrieved
    pub last_retrieved: Option<DateTime<Utc>>,
    /// Feature tags extracted from the problem for retrieval
    pub problem_features: Vec<String>,
    /// Whether this case has been validated by a human or judge
    pub validated: bool,
}

impl CaseMemory {
    /// Create a new case memory
    pub fn new(
        problem: impl Into<String>,
        plan: impl Into<String>,
        outcome: impl Into<String>,
        reward: f64,
    ) -> Self {
        Self {
            problem: problem.into(),
            plan: plan.into(),
            outcome: outcome.into(),
            reward: reward.clamp(0.0, 1.0),
            domain: None,
            retrieval_count: 0,
            last_retrieved: None,
            problem_features: Vec::new(),
            validated: false,
        }
    }

    /// Set the domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Add problem features
    pub fn with_features(mut self, features: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.problem_features
            .extend(features.into_iter().map(Into::into));
        self
    }

    /// Mark as validated
    pub fn validated(mut self) -> Self {
        self.validated = true;
        self
    }

    /// Record a retrieval of this case
    pub fn record_retrieval(&mut self) {
        self.retrieval_count += 1;
        self.last_retrieved = Some(Utc::now());
    }

    /// Whether this case represents a successful experience
    pub fn is_positive(&self) -> bool {
        self.reward >= 0.5
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let result_label = if self.is_positive() {
            "success"
        } else {
            "failure"
        };
        let mut parts = vec![
            format!("Problem: {}", self.problem),
            format!("Plan: {}", self.plan),
            format!("Outcome: {}", self.outcome),
            format!("Result: {}", result_label),
        ];

        if let Some(ref domain) = self.domain {
            parts.push(format!("Domain: {}", domain));
        }

        if !self.problem_features.is_empty() {
            parts.push(format!("Features: {}", self.problem_features.join(", ")));
        }

        parts.join("\n")
    }
}

/// An observation of a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternObservation {
    /// When the pattern was observed
    pub observed_at: DateTime<Utc>,
    /// Context of the observation
    pub context: String,
    /// Whether the pattern prediction was correct
    pub correct: bool,
}

impl PatternObservation {
    /// Create a new observation
    pub fn new(context: impl Into<String>, correct: bool) -> Self {
        Self {
            observed_at: Utc::now(),
            context: context.into(),
            correct,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_memory_creation() {
        let workflow = WorkflowMemory::new("deploy", "Deploy application to production")
            .with_triggers(vec!["deploy command", "CI/CD pipeline"])
            .with_step(StepDefinition::new(
                "build",
                "Build the application",
                "npm build",
            ))
            .with_step(StepDefinition::new("test", "Run tests", "npm test"))
            .with_step(StepDefinition::new(
                "deploy",
                "Deploy to server",
                "kubectl apply",
            ))
            .with_inputs(vec![WorkflowIO::required("version")])
            .with_outputs(vec![WorkflowIO::required("deployment_url")]);

        assert_eq!(workflow.steps.len(), 3);
        assert_eq!(workflow.triggers.len(), 2);
        assert_eq!(workflow.inputs.len(), 1);
    }

    #[test]
    fn workflow_execution_tracking() {
        let mut workflow = WorkflowMemory::new("test", "Test workflow");

        workflow.record_execution(true, Some(chrono::Duration::seconds(10)));
        assert_eq!(workflow.execution_count, 1);
        assert_eq!(workflow.success_rate, Some(1.0));

        workflow.record_execution(false, Some(chrono::Duration::seconds(5)));
        assert_eq!(workflow.execution_count, 2);
        assert_eq!(workflow.success_rate, Some(0.5));
    }

    #[test]
    fn skill_memory_creation() {
        let skill = SkillMemory::new("code_review", "Review code for issues")
            .with_category("development")
            .with_invocation("When asked to review code")
            .with_example(SkillExample::new(
                "Review this function",
                "Found 2 issues: ...",
            ))
            .with_proficiency(0.8);

        assert_eq!(skill.proficiency, 0.8);
        assert_eq!(skill.examples.len(), 1);
    }

    #[test]
    fn skill_usage_tracking() {
        let mut skill = SkillMemory::new("test", "Test skill").with_proficiency(0.5);

        skill.record_usage(true);
        assert!(skill.proficiency > 0.5);
        assert_eq!(skill.usage_count, 1);

        let prof_after_success = skill.proficiency;
        skill.record_usage(false);
        assert!(skill.proficiency < prof_after_success);
    }

    #[test]
    fn pattern_memory_creation() {
        let pattern = PatternMemory::new(
            "greeting",
            "User greeting pattern",
            PatternType::Conversation,
        )
        .with_indicators(vec!["hello", "hi", "good morning"])
        .with_response("Respond with a friendly greeting");

        assert_eq!(pattern.indicators.len(), 3);
        assert!(pattern.typical_response.is_some());
    }

    #[test]
    fn pattern_recognition_tracking() {
        let mut pattern = PatternMemory::new("test", "Test pattern", PatternType::Other);

        pattern.record_recognition(true);
        assert_eq!(pattern.recognition_count, 1);
        assert_eq!(pattern.accuracy, 1.0);

        pattern.record_recognition(false);
        assert_eq!(pattern.recognition_count, 2);
        assert_eq!(pattern.accuracy, 0.5);
    }

    #[test]
    fn case_memory_creation() {
        let case = CaseMemory::new(
            "How to deploy a service",
            "1. Build image 2. Push to registry 3. Apply k8s manifest",
            "Service deployed successfully",
            1.0,
        )
        .with_domain("devops")
        .with_features(vec!["deployment", "kubernetes"]);

        assert_eq!(case.reward, 1.0);
        assert!(case.is_positive());
        assert_eq!(case.domain, Some("devops".to_string()));
        assert_eq!(case.problem_features.len(), 2);
        assert!(!case.validated);
    }

    #[test]
    fn case_memory_reward_clamping() {
        let high = CaseMemory::new("p", "pl", "o", 1.5);
        assert_eq!(high.reward, 1.0);

        let low = CaseMemory::new("p", "pl", "o", -0.5);
        assert_eq!(low.reward, 0.0);
    }

    #[test]
    fn case_memory_positive_negative() {
        let positive = CaseMemory::new("p", "pl", "o", 0.8);
        assert!(positive.is_positive());

        let negative = CaseMemory::new("p", "pl", "o", 0.3);
        assert!(!negative.is_positive());

        let boundary = CaseMemory::new("p", "pl", "o", 0.5);
        assert!(boundary.is_positive());
    }

    #[test]
    fn case_memory_retrieval_tracking() {
        let mut case = CaseMemory::new("p", "pl", "o", 1.0);
        assert_eq!(case.retrieval_count, 0);
        assert!(case.last_retrieved.is_none());

        case.record_retrieval();
        assert_eq!(case.retrieval_count, 1);
        assert!(case.last_retrieved.is_some());
    }

    #[test]
    fn case_memory_text_content() {
        let case =
            CaseMemory::new("fix bug", "check logs", "bug fixed", 1.0).with_domain("debugging");

        let text = case.text_content();
        assert!(text.contains("Problem: fix bug"));
        assert!(text.contains("Plan: check logs"));
        assert!(text.contains("Outcome: bug fixed"));
        assert!(text.contains("Result: success"));
        assert!(text.contains("Domain: debugging"));
    }
}
