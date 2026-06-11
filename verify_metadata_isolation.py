import asyncio
import shutil
from pathlib import Path
from membrain import MembrainClient

async def main():
    storage_path = Path("./test_metadata_db")
    if storage_path.exists():
        shutil.rmtree(storage_path)

    config = {
        "storage": {
            "backend": "memscaledb",
            "path": str(storage_path),
            "indexed_metadata_keys": ["user_id"]
        }
    }

    # Initialize client without default scope, using shared database config
    client = MembrainClient(config=config)

    try:
        print("=== Step 1: Storing Alice's Facts with Explicit Metadata ===")
        # Alice's fact
        res_alice = await client.store_fact(
            "alice loves coding in Python and Rust",
            confidence=0.9,
            metadata={"user_id": "alice"}
        )
        print(f"Alice's Fact Stored. Success: {res_alice.success}, ID: {res_alice.id}")

        print("\n=== Step 2: Storing Bob's Facts with Explicit Metadata ===")
        # Bob's fact
        res_bob = await client.store_fact(
            "bob prefers coding in Go and TypeScript",
            confidence=0.9,
            metadata={"user_id": "bob"}
        )
        print(f"Bob's Fact Stored. Success: {res_bob.success}, ID: {res_bob.id}")

        print("\n=== Step 3: Searching with NO Filters ===")
        all_results = await client.search("coding", limit=10)
        print(f"Total memories found without filters: {len(all_results.memories)}")
        for m in all_results.memories:
            print(f" - Content: '{m.content}', Type: {m.memory_type}")

        print("\n=== Step 4: Searching with filter: user_id = alice ===")
        alice_results = await client.search(
            "coding", 
            limit=10, 
            filters={"metadata": {"user_id": "alice"}}
        )
        print(f"Memories found for Alice filter: {len(alice_results.memories)}")
        for m in alice_results.memories:
            print(f" - Content: '{m.content}'")
            # Assert to prove isolation mathematically
            assert "bob" not in m.content.lower(), "Bleed detected! Found Bob's memory in Alice's results."

        print("\n=== Step 5: Searching with filter: user_id = bob ===")
        bob_results = await client.search(
            "coding", 
            limit=10, 
            filters={"metadata": {"user_id": "bob"}}
        )
        print(f"Memories found for Bob filter: {len(bob_results.memories)}")
        for m in bob_results.memories:
            print(f" - Content: '{m.content}'")
            # Assert to prove isolation mathematically
            assert "alice" not in m.content.lower(), "Bleed detected! Found Alice's memory in Bob's results."

        print("\n🎉 SUCCESS: Explicit metadata isolation for facts works perfectly!")

    finally:
        client.close()
        if storage_path.exists():
            shutil.rmtree(storage_path)

if __name__ == "__main__":
    asyncio.run(main())
