"""
Test the greedy coloration algorithm dsatur, used to allocate
overlapping cells to different dimension on axis 0 (in NZYX format).
"""
from nahual.utils import dsatur

def test_triangle_graph():
    """Test 3-clique (triangle) requires 3 colors"""
    graph = [[1, 2], [0, 2], [0, 1]]
    colors = dsatur(graph)

    # Verify proper coloring (no adjacent vertices share color)
    for u in range(len(graph)):
        for v in graph[u]:
            assert colors[u] != colors[v]

    # Verify optimal coloring (3 colors required)
    assert len(set(colors)) == 3

def test_square_graph():
    """Test bipartite graph (square) requires 2 colors"""
    graph = [[1, 3], [0, 2], [1, 3], [0, 2]]
    colors = dsatur(graph)

    # Verify proper coloring
    for u in range(len(graph)):
        for v in graph[u]:
            assert colors[u] != colors[v]

    # Verify optimal coloring (2 colors sufficient)
    assert len(set(colors)) == 2

    # Verify bipartition pattern
    assert colors[0] == colors[2]
    assert colors[1] == colors[3]
    assert colors[0] != colors[1]
