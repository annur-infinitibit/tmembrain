/** MultiTenantIndex -- isolated vector indices per tenant. */

import { randomUUID } from "node:crypto";
import { MultiTenantIndex } from "membrain";

const DIMENSION = 128;

function embed(seed) {
  const out = [];
  let state = seed * 1000 + 1;
  for (let i = 0; i < DIMENSION; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    out.push((state / 0x7fffffff) * 2 - 1);
  }
  return out;
}

const index = new MultiTenantIndex({
  dimension: DIMENSION,
  indexType: "hnsw",
  maxTenants: 100,
  indexConfig: { m: 16, ef_construction: 100 },
});

try {
  const tenants = ["team-alpha", "team-beta", "team-gamma"];
  for (const tenant of tenants) index.createTenant(tenant);

  for (const tenant of tenants) {
    const offset = tenants.indexOf(tenant) * 100;
    for (let i = 0; i < 50; i++) index.add(tenant, randomUUID(), embed(offset + i));
  }

  const query = embed(0);
  console.log("Search per tenant (same query):");
  for (const tenant of tenants) {
    const results = index.search(tenant, query, 3);
    const scores = results.map((r) => r.score.toFixed(4));
    console.log(`  ${tenant}: ${index.tenantLen(tenant)} vectors, scores=[${scores}]`);
  }

  index.deleteTenant("team-gamma");
  console.log(`\nAfter deleting team-gamma: ${JSON.stringify(index.listTenants())}`);
} finally {
  index.close();
}
