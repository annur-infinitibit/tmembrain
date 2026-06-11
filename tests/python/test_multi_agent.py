"""
Test Multi-Agent Systems Cookbook Examples
Based on docs/cookbooks/multi-agent.mdx
"""
import os
import random
import pytest
from dotenv import load_dotenv
from openai import OpenAI

try:
    from membrain import MembrainClient, MembrainGraph
    MEMBRAIN_AVAILABLE = True
except ImportError:
    MEMBRAIN_AVAILABLE = False
    pytest.skip("Membrain not installed", allow_module_level=True)

# Load environment variables
load_dotenv()


class TestSharedKnowledgeBase:
    """Test shared knowledge base for multiple agents"""

    def test_shared_memory_between_agents(self):
        """Test that multiple agents can share the same memory"""
        shared_memory = MembrainClient()

        try:
            # Agent 1: Research agent stores findings
            shared_memory.store_fact("Quantum computing uses qubits", confidence=0.9)
            shared_memory.store_event("research", "Researched quantum computing")

            # Agent 2: Writer agent reads shared knowledge
            context = shared_memory.search("quantum", limit=5)
            assert len(context.memories) > 0

            # Verify quantum information is accessible
            content = "\n".join([m.content for m in context.memories])
            assert "quantum" in content.lower() or "qubit" in content.lower()

            # Agent 2 adds its own contribution
            shared_memory.store_task("Write article about quantum computing")

            # Both agents' contributions should be searchable
            all_memories = shared_memory.search("quantum computing", limit=10)
            assert len(all_memories.memories) >= 2

        finally:
            shared_memory.close()

    def test_multi_agent_collaboration_with_openai(self, openai_available):
        """Test multi-agent collaboration using OpenAI"""
        shared_memory = MembrainClient()
        llm = OpenAI()

        try:
            # Simulate research agent
            topic = "machine learning"
            response = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Give me 2 facts about {topic}"}],
                max_tokens=100
            )
            research_result = response.choices[0].message.content

            # Store research findings
            shared_memory.store_fact(research_result, confidence=0.85)
            shared_memory.store_event("research", f"Researched {topic}")

            # Simulate writer agent retrieving context
            context = shared_memory.search(topic, limit=5)
            assert len(context.memories) > 0

            context_text = "\n".join([m.content for m in context.memories])
            assert len(context_text) > 0

        finally:
            shared_memory.close()


class TestAgentSkillRegistry:
    """Test agent skill registry pattern"""

    def test_register_agent_skills(self):
        """Test registering skills for multiple agents"""
        client = MembrainClient()

        try:
            # Register different agent skills
            agents = {
                "CodeReviewBot": [
                    ("code_analysis", "Analyzes code for bugs and style issues"),
                    ("security_scan", "Scans for security vulnerabilities"),
                ],
                "TestBot": [
                    ("unit_test_gen", "Generates unit tests"),
                    ("integration_test", "Creates integration tests"),
                ],
                "DocBot": [
                    ("api_docs", "Generates API documentation"),
                    ("tutorial_writer", "Writes tutorials"),
                ],
            }

            skill_count = 0
            for agent_name, skills in agents.items():
                for skill_name, description in skills:
                    client.store_skill(skill_name, description)
                    client.store_entity(agent_name, "agent")
                    skill_count += 1

            # Find appropriate agent for a task
            task = "need code review"
            results = client.search(task, limit=3)
            assert len(results.memories) > 0

        finally:
            client.close()

    def test_skill_discovery(self):
        """Test discovering skills based on task requirements"""
        client = MembrainClient()

        try:
            # Register skills
            client.store_skill("data_processing", "Processes large datasets efficiently")
            client.store_skill("visualization", "Creates charts and graphs")
            client.store_skill("reporting", "Generates analytical reports")

            # Search for relevant skill
            results = client.search("need to create charts", limit=3)
            assert len(results.memories) > 0

            # Verify visualization skill is found
            skills = [m.content for m in results.memories]
            assert any("visual" in s.lower() or "chart" in s.lower() for s in skills)

        finally:
            client.close()


class TestAgentCoordination:
    """Test multi-agent coordination through shared memory"""

    def test_agent_class_communication(self, tmp_path):
        """Test Agent class for communication through memory"""
        config = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "agent_comms")
            }
        }

        class Agent:
            def __init__(self, name, memory_client):
                self.name = name
                self.client = memory_client
                self.client.store_entity(self.name, "agent")

            def log_action(self, action, details):
                self.client.store_event(
                    f"{self.name}_{action}",
                    f"{self.name} {details}"
                )

            def communicate(self, message):
                self.client.store_observation(
                    f"{self.name} says {message}"
                )

            def get_team_context(self, search_term):
                results = self.client.search(search_term, limit=10)
                return [m.content for m in results.memories]

        client = MembrainClient(config=config)

        try:
            # Create team
            planner = Agent("Planner", client)
            executor = Agent("Executor", client)
            reviewer = Agent("Reviewer", client)

            # Workflow
            planner.communicate("Starting project planning for the team")
            planner.log_action("plan", "Created 5 tasks for team")

            # Executor reads context - search for stored observations
            context = executor.get_team_context("Planner")
            assert len(context) > 0
            assert any("Planner" in msg for msg in context)

            # Executor works
            executor.communicate("Executing task 1 in progress")
            executor.log_action("execute", "Completed task 1 successfully")

            # Reviewer checks
            reviewer.communicate("Reviewing work status now")
            review_context = reviewer.get_team_context("task")
            # At least some messages should be found
            assert len(review_context) >= 1

        finally:
            client.close()

    def test_agent_workflow(self):
        """Test complete agent workflow with task delegation"""
        client = MembrainClient()

        try:
            # Store workflow steps
            client.store_task("Agent1: Analyze requirements")
            client.store_task("Agent2: Implement solution")
            client.store_task("Agent3: Test implementation")
            client.store_task("Agent4: Deploy to production")

            # Simulate agent picking up tasks
            agent1_tasks = client.search("Agent1", limit=5)
            assert len(agent1_tasks.memories) > 0

            # Mark task as done
            client.store_event("task_complete", "Agent1 completed requirement analysis")

            # Agent2 gets context
            context = client.search("requirement analysis", limit=5)
            assert len(context.memories) > 0

        finally:
            client.close()


class TestAgentState:
    """Test agent state tracking"""

    def test_stateful_agent(self, tmp_path):
        """Test tracking agent goals and tasks"""
        config = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "stateful_agent")
            }
        }

        class StatefulAgent:
            def __init__(self, agent_id, client):
                self.agent_id = agent_id
                self.client = client

            def update_goal(self, goal):
                result = self.client.store_goal(f"{self.agent_id} goal {goal}")
                return result.id if result.success else None

            def add_task(self, task):
                self.client.store_task(f"{self.agent_id} task {task}")

            def learn_pattern(self, pattern_name, description, ptype):
                self.client.store_pattern(pattern_name, description, ptype)

            def get_state(self):
                goals = self.client.search(f"{self.agent_id} goal", limit=5)
                tasks = self.client.search(f"{self.agent_id} task", limit=5)
                return {
                    "goals": [m.content for m in goals.memories],
                    "tasks": [m.content for m in tasks.memories],
                }

        client = MembrainClient(config=config)

        try:
            agent = StatefulAgent("Agent-001", client)

            # Set goals and tasks
            goal_id = agent.update_goal("Improve code quality in all modules")
            assert goal_id is not None

            agent.add_task("Run linter on all source files")
            agent.add_task("Fix type annotation errors throughout codebase")
            agent.learn_pattern("singleton", "One instance design pattern", "design")

            # Get agent state
            state = agent.get_state()
            assert "goals" in state
            assert "tasks" in state
            assert len(state["goals"]) > 0
            assert len(state["tasks"]) >= 1

        finally:
            client.close()

    def test_multiple_agent_states(self):
        """Test tracking state for multiple agents"""
        client = MembrainClient()

        try:
            # Agent 1 state
            client.store_goal("Agent1: Complete data migration")
            client.store_task("Agent1: Backup database")
            client.store_task("Agent1: Run migration script")

            # Agent 2 state
            client.store_goal("Agent2: Improve test coverage")
            client.store_task("Agent2: Write unit tests")

            # Retrieve Agent 1 state
            agent1_goals = client.search("Agent1 goal", limit=5)
            agent1_tasks = client.search("Agent1 task", limit=5)

            assert len(agent1_goals.memories) >= 1
            assert len(agent1_tasks.memories) >= 2

            # Retrieve Agent 2 state
            agent2_goals = client.search("Agent2 goal", limit=5)
            assert len(agent2_goals.memories) >= 1

        finally:
            client.close()


class TestHierarchicalAgents:
    """Test hierarchical agent systems"""

    def test_manager_worker_pattern(self):
        """Test manager agent coordinating worker agents"""
        class ManagerAgent:
            def __init__(self, name, client):
                self.name = name
                self.client = client
                self.workers = []

            def assign_task(self, worker, task):
                self.client.store_task(f"{worker}: {task}")
                self.client.store_event("task_assigned",
                    f"{self.name} assigned task to {worker}")

            def get_team_status(self):
                results = self.client.search(f"task assigned {self.name}", limit=20)
                return results.memories

        client = MembrainClient()

        try:
            manager = ManagerAgent("ManagerBot", client)

            # Assign tasks to workers
            manager.assign_task("WorkerBot1", "Process data pipeline")
            manager.assign_task("WorkerBot2", "Generate reports")
            manager.assign_task("WorkerBot3", "Send notifications")

            # Check team status
            status = manager.get_team_status()
            assert len(status) >= 3

        finally:
            client.close()


class TestCollaborativeLearning:
    """Test collaborative learning between agents"""

    def test_shared_knowledge_graph(self, tmp_path):
        """Test multiple agents contributing to shared knowledge graph"""
        config = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "knowledge_graph")
            }
        }
        client = MembrainClient(config=config)
        graph = MembrainGraph(config={"embedding_dim": 16})

        try:
            agents = ["Agent-Alpha", "Agent-Beta", "Agent-Gamma"]

            # Each agent contributes distinct observations
            all_ids = []
            for agent in agents:
                for i in range(3):
                    result = client.store_observation(
                        f"{agent} discovered unique insight number {i} about the world"
                    )
                    # Only add to graph if stored successfully
                    if result.success and result.id is not None:
                        all_ids.append(result.id)
                        emb = [random.random() for _ in range(16)]
                        graph.add_node(result.id, emb)

            # Verify at least some observations are in graph
            assert len(all_ids) > 0
            assert graph.node_count() == len(all_ids)

            # Any agent can query collective knowledge
            query = "discovered insight"
            results = client.search(query, limit=20)
            assert len(results.memories) > 0

        finally:
            client.close()
            graph.close()

    def test_agent_learning_from_others(self):
        """Test agents learning from each other's experiences"""
        client = MembrainClient()

        try:
            # Agent A learns something
            client.store_observation("AgentA learned: Python functions can return multiple values")

            # Agent B learns something else (explicitly mentions Python so
            # text-only search without embeddings can still match it)
            client.store_observation("AgentB learned: Python list comprehensions are faster than loops")

            # Agent C queries collective knowledge
            python_knowledge = client.search("Python", limit=10)
            assert len(python_knowledge.memories) >= 2

            # Verify both learnings are accessible
            content = "\n".join([m.content for m in python_knowledge.memories])
            assert "function" in content.lower() or "comprehension" in content.lower()

        finally:
            client.close()


class TestAgentCommunication:
    """Test communication patterns between agents"""

    def test_broadcast_message(self):
        """Test broadcasting messages to all agents"""
        client = MembrainClient()

        try:
            # Broadcast message
            client.store_event("broadcast", "System maintenance scheduled for tonight")

            # All agents can receive broadcast
            for agent_id in ["Agent1", "Agent2", "Agent3"]:
                messages = client.search("broadcast maintenance", limit=5)
                assert len(messages.memories) > 0
                assert "maintenance" in messages.memories[0].content.lower()

        finally:
            client.close()

    def test_direct_message(self):
        """Test direct messaging between specific agents"""
        client = MembrainClient()

        try:
            # Agent1 sends message to Agent2
            client.store_observation("Agent1 to Agent2: Please review PR #123")

            # Agent2 retrieves message
            messages = client.search("Agent1 to Agent2", limit=5)
            assert len(messages.memories) > 0
            assert "PR" in messages.memories[0].content

        finally:
            client.close()


class TestAgentPreferences:
    """Test storing and retrieving agent preferences"""

    def test_agent_preferences(self):
        """Test storing preferences for different agents"""
        client = MembrainClient()

        try:
            # Store agent preferences
            client.store_preference("AgentA", "model", "prefers GPT-4", "strong")
            client.store_preference("AgentA", "style", "verbose responses", "moderate")
            client.store_preference("AgentB", "model", "prefers Claude", "strong")

            # Retrieve agent preferences
            agent_a_prefs = client.search("AgentA preferences", limit=5)
            assert len(agent_a_prefs.memories) >= 2

        finally:
            client.close()


class TestTeamMetrics:
    """Test tracking metrics for agent teams"""

    def test_team_activity_metrics(self, tmp_path):
        """Test tracking team activity through events"""
        config = {
            "storage": {
                "backend": "memscaledb",
                "path": str(tmp_path / "team_metrics")
            }
        }
        client = MembrainClient(config=config)

        try:
            # Simulate team activity with distinct content
            agents = ["Agent1", "Agent2", "Agent3"]
            stored_count = 0
            for agent in agents:
                r1 = client.store_event(
                    "task_complete",
                    f"{agent} completed requirement analysis for sprint"
                )
                r2 = client.store_event(
                    "message_sent",
                    f"{agent} sent status update to project manager"
                )
                if r1.success:
                    stored_count += 1
                if r2.success:
                    stored_count += 1

            # Verify some events were stored
            assert stored_count > 0

            # Get overall stats
            stats = client.stats()
            assert stats["total_memories"] >= 1

        finally:
            client.close()
