import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { MultiTenantIndex } from "../dist/index.js";

const DIMENSION = 16;

function randomVector(dimension = DIMENSION) {
  return Array.from({ length: dimension }, () => Math.random());
}

describe("MultiTenantIndex", () => {
  it("should create and list tenants", () => {
    const index = new MultiTenantIndex({ dimension: DIMENSION, indexType: "flat" });
    index.createTenant("alice");
    index.createTenant("bob");

    assert.equal(index.tenantCount(), 2);
    assert.ok(index.hasTenant("alice"));
    assert.ok(index.hasTenant("bob"));
    assert.ok(!index.hasTenant("charlie"));

    const tenants = index.listTenants();
    assert.deepEqual(tenants, ["alice", "bob"]);

    index.close();
  });

  it("should delete a tenant", () => {
    const index = new MultiTenantIndex({ dimension: DIMENSION, indexType: "flat" });
    index.createTenant("alice");
    assert.ok(index.hasTenant("alice"));

    const found = index.deleteTenant("alice");
    assert.equal(found, true);
    assert.ok(!index.hasTenant("alice"));

    const notFound = index.deleteTenant("alice");
    assert.equal(notFound, false);

    index.close();
  });

  it("should add and search per tenant", () => {
    const index = new MultiTenantIndex({ dimension: 3, indexType: "flat" });
    index.createTenant("alice");

    const vecId = randomUUID();
    index.add("alice", vecId, [1.0, 0.0, 0.0]);
    assert.equal(index.tenantLen("alice"), 1);

    const results = index.search("alice", [1.0, 0.1, 0.0], 1);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, vecId);

    index.close();
  });

  it("should isolate tenants from each other", () => {
    const index = new MultiTenantIndex({ dimension: 3, indexType: "flat" });
    index.createTenant("alice");
    index.createTenant("bob");

    index.add("alice", randomUUID(), [1.0, 0.0, 0.0]);

    const aliceResults = index.search("alice", [1.0, 0.0, 0.0], 1);
    assert.equal(aliceResults.length, 1);

    const bobResults = index.search("bob", [1.0, 0.0, 0.0], 1);
    assert.equal(bobResults.length, 0);

    index.close();
  });

  it("should enforce max_tenants limit", () => {
    const index = new MultiTenantIndex({
      dimension: DIMENSION, indexType: "flat", maxTenants: 2,
    });
    index.createTenant("a");
    index.createTenant("b");

    assert.throws(() => index.createTenant("c"));

    index.close();
  });

  it("should remove a vector from a tenant", () => {
    const index = new MultiTenantIndex({ dimension: 3, indexType: "flat" });
    index.createTenant("alice");

    const vecId = randomUUID();
    index.add("alice", vecId, [1.0, 0.0, 0.0]);
    assert.equal(index.tenantLen("alice"), 1);

    const found = index.remove("alice", vecId);
    assert.equal(found, true);
    assert.equal(index.tenantLen("alice"), 0);

    index.close();
  });

  it("should report correct dimension", () => {
    const index = new MultiTenantIndex({ dimension: 768, indexType: "flat" });
    assert.equal(index.dimension(), 768);
    index.close();
  });

  it("should work with hnsw index type", () => {
    const index = new MultiTenantIndex({ dimension: 8, indexType: "hnsw" });
    index.createTenant("alice");

    const vecId = randomUUID();
    index.add("alice", vecId, Array(8).fill(0.5));

    const results = index.search("alice", Array(8).fill(0.5), 1);
    assert.equal(results.length, 1);
    assert.equal(results[0].id, vecId);

    index.close();
  });
});
