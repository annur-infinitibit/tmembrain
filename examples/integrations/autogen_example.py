"""
AutoGen + Membrain Integration Example

This shows how to use Membrain with AutoGen for multi-agent conversations.
Membrain tracks agent interactions and learned behaviors.
"""

from membrain import MembrainClient

# Initialize Membrain
memory = MembrainClient()


class MemoryAgent:
    """Simple agent with Membrain memory"""

    def __init__(self, name: str):
        self.name = name
        self.memory = MembrainClient()

    def send_message(self, recipient: str, message: str):
        """Send a message and store in memory"""
        self.memory.store_observation(f"{self.name} to {recipient}: {message}")
        print(f"[{self.name}] → {message}")

    def receive_message(self, sender: str, message: str):
        """Receive and store a message"""
        self.memory.store_observation(f"{sender} to {self.name}: {message}")
        print(f"[{sender}] → {self.name}: {message}")

    def recall(self, query: str, limit: int = 3):
        """Recall relevant memories"""
        results = self.memory.search(query, limit=limit)
        return [m.content for m in results.memories]

    def learn_skill(self, skill_name: str, description: str):
        """Record a learned skill"""
        self.memory.store_skill(skill_name, description)
        print(f"[{self.name}] Learned: {skill_name}")


# Example usage
if __name__ == "__main__":
    print("AutoGen + Membrain Agent Memory\n")

    # Create agents
    agent1 = MemoryAgent("Researcher")
    agent2 = MemoryAgent("Writer")

    # Agent interaction
    agent1.send_message("Writer", "I found great info about Python!")
    agent2.receive_message("Researcher", "I found great info about Python!")

    agent2.send_message("Researcher", "Great! I'll write an article about it.")
    agent1.receive_message("Writer", "Great! I'll write an article about it.")

    # Learn skills
    agent1.learn_skill("research", "Finding relevant information on topics")
    agent2.learn_skill("writing", "Creating engaging technical articles")

    # Recall
    print(f"\n{agent1.name}'s memories about Python:")
    for mem in agent1.recall("Python"):
        print(f"  - {mem}")

    print(f"\nTotal memories: {agent1.memory.count()}")
