//! Agent state memory types: goals, tasks, working memory
//!
//! Agent state memories represent the current context and intentions
//! of an agent. These are typically short-lived and frequently updated.

use super::MemoryCommon;
use crate::types::MemoryId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Agent state memory containing current context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStateMemory {
    /// Common memory fields
    pub common: MemoryCommon,
    /// The specific agent state content
    pub content: AgentStateContent,
}

impl AgentStateMemory {
    /// Get text content for embedding/indexing
    pub fn text_content(&self) -> String {
        match &self.content {
            AgentStateContent::Goal(g) => g.text_content(),
            AgentStateContent::Task(t) => t.text_content(),
            AgentStateContent::WorkingMemory(w) => w.text_content(),
        }
    }
}

/// Types of agent state content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "subtype", content = "data")]
pub enum AgentStateContent {
    /// A goal the agent is working toward
    Goal(Goal),
    /// A task to be completed
    Task(Task),
    /// Working memory item
    WorkingMemory(WorkingMemoryItem),
}

/// A goal the agent is working toward
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Description of the goal
    pub description: String,
    /// Current status
    pub status: GoalStatus,
    /// Priority (0.0-1.0)
    pub priority: f64,
    /// Parent goal (if this is a sub-goal)
    pub parent_goal: Option<MemoryId>,
    /// Sub-goals that contribute to this goal
    pub sub_goals: Vec<MemoryId>,
    /// Tasks associated with this goal
    pub tasks: Vec<MemoryId>,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Progress (0.0-1.0)
    pub progress: f64,
    /// Deadline (if any)
    pub deadline: Option<DateTime<Utc>>,
    /// When this goal was created
    pub created_at: DateTime<Utc>,
    /// When this goal was completed (if completed)
    pub completed_at: Option<DateTime<Utc>>,
    /// Reason for current status (especially for blocked/failed)
    pub status_reason: Option<String>,
}

impl Goal {
    /// Create a new goal
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            status: GoalStatus::Active,
            priority: 0.5,
            parent_goal: None,
            sub_goals: Vec::new(),
            tasks: Vec::new(),
            success_criteria: Vec::new(),
            progress: 0.0,
            deadline: None,
            created_at: Utc::now(),
            completed_at: None,
            status_reason: None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Set parent goal
    pub fn with_parent(mut self, parent: MemoryId) -> Self {
        self.parent_goal = Some(parent);
        self
    }

    /// Add success criteria
    pub fn with_criteria(mut self, criteria: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.success_criteria
            .extend(criteria.into_iter().map(Into::into));
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add a sub-goal
    pub fn add_sub_goal(&mut self, sub_goal_id: MemoryId) {
        self.sub_goals.push(sub_goal_id);
    }

    /// Add a task
    pub fn add_task(&mut self, task_id: MemoryId) {
        self.tasks.push(task_id);
    }

    /// Update progress
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 1.0);
        if self.progress >= 1.0 && self.status == GoalStatus::Active {
            self.complete();
        }
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        self.status = GoalStatus::Completed;
        self.progress = 1.0;
        self.completed_at = Some(Utc::now());
    }

    /// Mark as blocked
    pub fn block(&mut self, reason: impl Into<String>) {
        self.status = GoalStatus::Blocked;
        self.status_reason = Some(reason.into());
    }

    /// Mark as failed
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = GoalStatus::Failed;
        self.status_reason = Some(reason.into());
    }

    /// Check if goal is overdue
    pub fn is_overdue(&self) -> bool {
        if let Some(deadline) = self.deadline {
            self.status == GoalStatus::Active && Utc::now() > deadline
        } else {
            false
        }
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![
            format!("Goal: {}", self.description),
            format!("Status: {:?}", self.status),
            format!("Progress: {:.0}%", self.progress * 100.0),
        ];

        if !self.success_criteria.is_empty() {
            parts.push(format!("Criteria: {}", self.success_criteria.join(", ")));
        }

        if let Some(ref reason) = self.status_reason {
            parts.push(format!("Note: {}", reason));
        }

        parts.join("\n")
    }
}

/// Status of a goal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GoalStatus {
    /// Goal is being actively worked on
    Active,
    /// Goal is paused
    Paused,
    /// Goal is blocked by dependencies
    Blocked,
    /// Goal has been completed successfully
    Completed,
    /// Goal failed
    Failed,
    /// Goal was abandoned
    Abandoned,
}

/// A task to be completed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Title of the task
    pub title: String,
    /// Description of what needs to be done
    pub description: Option<String>,
    /// Current status
    pub status: TaskStatus,
    /// Priority (0.0-1.0)
    pub priority: f64,
    /// Goal this task belongs to
    pub goal_id: Option<MemoryId>,
    /// Tasks that must complete before this one
    pub dependencies: Vec<MemoryId>,
    /// Tasks blocked by this one
    pub blocking: Vec<MemoryId>,
    /// Due date (if any)
    pub due_date: Option<DateTime<Utc>>,
    /// Estimated effort (in minutes)
    pub estimated_effort: Option<u32>,
    /// Actual effort spent (in minutes)
    pub actual_effort: Option<u32>,
    /// When this task was created
    pub created_at: DateTime<Utc>,
    /// When this task was started
    pub started_at: Option<DateTime<Utc>>,
    /// When this task was completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Result/outcome of the task
    pub result: Option<String>,
    /// Labels/tags
    pub labels: Vec<String>,
}

impl Task {
    /// Create a new task
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: None,
            status: TaskStatus::Pending,
            priority: 0.5,
            goal_id: None,
            dependencies: Vec::new(),
            blocking: Vec::new(),
            due_date: None,
            estimated_effort: None,
            actual_effort: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            result: None,
            labels: Vec::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Set goal
    pub fn with_goal(mut self, goal_id: MemoryId) -> Self {
        self.goal_id = Some(goal_id);
        self
    }

    /// Add a dependency
    pub fn with_dependency(mut self, dep_id: MemoryId) -> Self {
        self.dependencies.push(dep_id);
        self
    }

    /// Set due date
    pub fn with_due_date(mut self, due: DateTime<Utc>) -> Self {
        self.due_date = Some(due);
        self
    }

    /// Set estimated effort
    pub fn with_estimated_effort(mut self, minutes: u32) -> Self {
        self.estimated_effort = Some(minutes);
        self
    }

    /// Add labels
    pub fn with_labels(mut self, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.labels.extend(labels.into_iter().map(Into::into));
        self
    }

    /// Start the task
    pub fn start(&mut self) {
        self.status = TaskStatus::InProgress;
        self.started_at = Some(Utc::now());
    }

    /// Complete the task
    pub fn complete(&mut self, result: Option<String>) {
        self.status = TaskStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.result = result;

        // Calculate actual effort if we have start time
        if let Some(started) = self.started_at {
            let duration = Utc::now() - started;
            self.actual_effort = Some(duration.num_minutes() as u32);
        }
    }

    /// Mark as failed
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = TaskStatus::Failed;
        self.result = Some(reason.into());
    }

    /// Mark as blocked
    pub fn block(&mut self) {
        self.status = TaskStatus::Blocked;
    }

    /// Check if task is overdue
    pub fn is_overdue(&self) -> bool {
        if let Some(due) = self.due_date {
            self.status != TaskStatus::Completed && Utc::now() > due
        } else {
            false
        }
    }

    /// Check if all dependencies are met
    pub fn dependencies_met(&self, completed_tasks: &[MemoryId]) -> bool {
        self.dependencies
            .iter()
            .all(|d| completed_tasks.contains(d))
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![
            format!("Task: {}", self.title),
            format!("Status: {:?}", self.status),
        ];

        if let Some(ref desc) = self.description {
            parts.push(format!("Description: {}", desc));
        }

        if !self.labels.is_empty() {
            parts.push(format!("Labels: {}", self.labels.join(", ")));
        }

        if let Some(ref result) = self.result {
            parts.push(format!("Result: {}", result));
        }

        parts.join("\n")
    }
}

/// Status of a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// Task is waiting to be started
    Pending,
    /// Task is currently being worked on
    InProgress,
    /// Task is blocked by dependencies
    Blocked,
    /// Task has been completed
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
}

/// An item in working memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryItem {
    /// Type of working memory item
    pub item_type: WorkingMemoryType,
    /// Content of the item
    pub content: String,
    /// Additional structured data
    pub data: Option<serde_json::Value>,
    /// Relevance score (0.0-1.0)
    pub relevance: f64,
    /// When this was added to working memory
    pub added_at: DateTime<Utc>,
    /// Expiration time
    pub expires_at: Option<DateTime<Utc>>,
    /// Source of this information
    pub source: Option<String>,
    /// Related memories
    pub related: Vec<MemoryId>,
}

impl WorkingMemoryItem {
    /// Create a new working memory item
    pub fn new(item_type: WorkingMemoryType, content: impl Into<String>) -> Self {
        Self {
            item_type,
            content: content.into(),
            data: None,
            relevance: 0.5,
            added_at: Utc::now(),
            expires_at: None,
            source: None,
            related: Vec::new(),
        }
    }

    /// Set structured data
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Set relevance
    pub fn with_relevance(mut self, relevance: f64) -> Self {
        self.relevance = relevance.clamp(0.0, 1.0);
        self
    }

    /// Set expiration
    pub fn with_expiration(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add related memory
    pub fn with_related(mut self, memory_id: MemoryId) -> Self {
        self.related.push(memory_id);
        self
    }

    /// Check if expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|exp| Utc::now() > exp)
    }

    /// Get text content for embedding
    pub fn text_content(&self) -> String {
        let mut parts = vec![format!(
            "Working Memory ({:?}): {}",
            self.item_type, self.content
        )];

        if let Some(ref source) = self.source {
            parts.push(format!("Source: {}", source));
        }

        parts.join("\n")
    }
}

/// Type of working memory item
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkingMemoryType {
    /// Current context information
    Context,
    /// Instruction or directive
    Instruction,
    /// Temporary fact
    TemporaryFact,
    /// Reference to another memory
    Reference,
    /// Scratchpad/notes
    Scratchpad,
    /// Error or warning
    Alert,
    /// User preference for current session
    SessionPreference,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_creation_and_progress() {
        let mut goal = Goal::new("Complete project")
            .with_priority(0.8)
            .with_criteria(vec!["All tasks done", "Tests passing"]);

        assert_eq!(goal.status, GoalStatus::Active);
        assert_eq!(goal.progress, 0.0);

        goal.update_progress(0.5);
        assert_eq!(goal.progress, 0.5);

        goal.update_progress(1.0);
        assert_eq!(goal.status, GoalStatus::Completed);
        assert!(goal.completed_at.is_some());
    }

    #[test]
    fn goal_blocking() {
        let mut goal = Goal::new("Test goal");
        goal.block("Waiting for dependencies");

        assert_eq!(goal.status, GoalStatus::Blocked);
        assert_eq!(
            goal.status_reason,
            Some("Waiting for dependencies".to_string())
        );
    }

    #[test]
    fn task_lifecycle() {
        let mut task = Task::new("Write tests")
            .with_description("Add unit tests")
            .with_priority(0.9)
            .with_labels(vec!["testing", "urgent"]);

        assert_eq!(task.status, TaskStatus::Pending);

        task.start();
        assert_eq!(task.status, TaskStatus::InProgress);
        assert!(task.started_at.is_some());

        task.complete(Some("All tests written".to_string()));
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(task.completed_at.is_some());
        assert!(task.actual_effort.is_some());
    }

    #[test]
    fn task_dependencies() {
        let task = Task::new("Deploy")
            .with_dependency(MemoryId::new())
            .with_dependency(MemoryId::new());

        assert_eq!(task.dependencies.len(), 2);
        assert!(!task.dependencies_met(&[]));
    }

    #[test]
    fn working_memory_creation() {
        let item = WorkingMemoryItem::new(WorkingMemoryType::Context, "User is asking about Rust")
            .with_relevance(0.9)
            .with_source("conversation")
            .with_expiration(Utc::now() + chrono::Duration::hours(1));

        assert_eq!(item.item_type, WorkingMemoryType::Context);
        assert!(!item.is_expired());
    }

    #[test]
    fn working_memory_expiration() {
        let item = WorkingMemoryItem::new(WorkingMemoryType::TemporaryFact, "temp")
            .with_expiration(Utc::now() - chrono::Duration::hours(1));

        assert!(item.is_expired());
    }
}
