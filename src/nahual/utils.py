def dsatur(graph:list[list[int]])->list[int]:
    n = len(graph)
    degree = [len(graph[i]) for i in range(n)]  # Original vertex degrees
    color = [-1] * n  # -1 = uncolored

    # Step 1: Color vertex with highest degree
    v0 = max(range(n), key=lambda v: degree[v])
    color[v0] = 0

    # While uncolored vertices exist
    while -1 in color:
        # Calculate saturation degrees (number of distinct colored neighbors)
        saturation = [-1] * n
        for v in range(n):
            if color[v] != -1:
                continue
            saturation[v] = len({color[u] for u in graph[v] if color[u] != -1})

        # Select next vertex: max saturation → max degree tiebreaker
        candidate = -1
        for v in range(n):
            if color[v] == -1:
                if (candidate == -1 or 
                    saturation[v] > saturation[candidate] or 
                    (saturation[v] == saturation[candidate] and degree[v] > degree[candidate])):
                    candidate = v

        # Assign smallest available color
        used = {color[u] for u in graph[candidate] if color[u] != -1}
        color[candidate] = next(c for c in range(n) if c not in used)

    return color
