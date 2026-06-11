import pytest
from membrain import MembrainGraph

async def test_graph_query_empty():
    with MembrainGraph() as graph:
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

async def test_graph_closed_raises():
    graph = MembrainGraph()
    graph.close()
    with pytest.raises(RuntimeError, match="graph is closed"):
        graph.node_count()

async def test_add_query_roundtrip():
    import uuid
    with MembrainGraph(config={"embedding_dim": 3}) as graph:
        mid1 = str(uuid.uuid4())
        mid2 = str(uuid.uuid4())
        graph.add_node(mid1, [1.0, 0.0, 0.0], confidence=0.8)
        graph.add_node(mid2, [0.0, 1.0, 0.0], confidence=0.9)
        assert graph.node_count() == 2
        
        res = await graph.query([1.0, 0.0, 0.0], top_k=2)
        assert len(res.nodes) > 0
        assert res.nodes[0].memory_id == mid1

async def test_save_load():
    import uuid
    with MembrainGraph(config={"embedding_dim": 3}) as graph:
        mid = str(uuid.uuid4())
        graph.add_node(mid, [0.5, 0.5, 0.5])
        assert graph.node_count() == 1
        
        data = graph.save()
        assert len(data) > 0
        
    # Load into a new graph
    with MembrainGraph.load(data) as loaded_graph:
        assert loaded_graph.node_count() == 1
