"""MultiTenantIndex -- isolated vector indices per tenant."""

import random
import uuid

from membrain import MultiTenantIndex

DIMENSION = 128


def embed(seed: float) -> list[float]:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(DIMENSION)]


def main() -> None:
    with MultiTenantIndex(
        dimension=DIMENSION, index_type="hnsw", max_tenants=100,
        index_config={"m": 16, "ef_construction": 100},
    ) as index:
        tenants = ["team-alpha", "team-beta", "team-gamma"]
        for tenant in tenants:
            index.create_tenant(tenant)

        for tenant in tenants:
            offset = tenants.index(tenant) * 100
            for i in range(50):
                index.add(tenant, str(uuid.uuid4()), embed(seed=float(offset + i)))

        query = embed(seed=0.0)
        print("Search per tenant (same query):")
        for tenant in tenants:
            results = index.search(tenant, query, k=3)
            scores = [f"{r.score:.4f}" for r in results]
            print(f"  {tenant}: {index.tenant_len(tenant)} vectors, scores={scores}")

        index.delete_tenant("team-gamma")
        print(f"\nAfter deleting team-gamma: {index.list_tenants()}")


if __name__ == "__main__":
    main()
