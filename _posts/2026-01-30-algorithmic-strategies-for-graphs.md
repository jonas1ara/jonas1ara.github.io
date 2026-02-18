---
title: Algorithmic Strategies for Graphs 🕸️
description: Master the most common algorithmic strategies to solve graph problems in LeetCode. Learn how to identify patterns and apply the right approach with real C# implementations.
Author: Jonas Lara
date: 2026-01-30 00:00:00 +0000
categories: [Algorithms, Data Structures, LeetCode]
tags: [graphs, dfs, bfs, union find, topological sort, dijkstra, connected components]
image:
  path: /assets/img/post/algorithmic-strategies-for-graphs/graphs-kp.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/algorithmic-strategies-for-graphs/graphs-kp.png
  alt: Graph Data Structure Visualization
---

## Introduction

Graphs are versatile data structures that model relationships between entities, consisting of vertices (nodes) and edges (connections). **The key to solving graph problems efficiently is recognizing patterns and applying the right algorithmic strategy**. This guide explores the most common strategies with real C# implementations from LeetCode problems.

## Common Algorithmic Strategies

When approaching graph problems, you'll typically encounter one of these strategies:

- **Depth-First Search (DFS)**: Explore as deep as possible before backtracking
- **Breadth-First Search (BFS)**: Explore layer by layer using a queue
- **Union Find (Disjoint Set)**: Track and merge connected components efficiently
- **Topological Sort**: Order vertices in a directed acyclic graph (DAG)
- **Dijkstra's Algorithm**: Find shortest paths from a source vertex
- **Cycle Detection**: Identify cycles in directed or undirected graphs
- **Graph Coloring**: Assign colors to vertices with constraints
- **Brute Force**: Try all possible paths or combinations

## Strategy 1: Depth-First Search (DFS)

**Depth-First Search (DFS)** is a traversal algorithm that explores as far as possible along each branch before backtracking. For graphs, DFS can be implemented recursively (using the call stack) or iteratively (using an explicit stack). DFS is ideal for exploring all paths, detecting cycles, finding connected components, and solving maze-like problems. It uses O(V) space for the visited set and O(V) space for the recursion stack in the worst case.

### When to use it

- Detect cycles in graphs
- Find connected components
- Path finding problems
- Topological sorting
- Solve maze problems
- Explore all possibilities

### Time Complexity: O(V + E) | Space Complexity: O(V)

### Problem: Number of Islands

Count the number of islands in a 2D grid (1 = land, 0 = water).

**Implementation:**

```csharp
public class Solution
{
    private int m, n;
    private int[][] dirs = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 } };

    private void dfs(char[][] grid, int x, int y)
    {
        if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] != '1') 
        {
            return;
        }

        grid[x][y] = 'x'; // Mark as visited

        foreach (var dir in dirs)
        {
            dfs(grid, x + dir[0], y + dir[1]);
        }
    }

    public int NumIslands(char[][] grid)
    {
        if (grid == null || grid.Length == 0 || grid[0].Length == 0) 
        {   
            return 0;
        }

        int ans = 0;
        m = grid.Length;
        n = grid[0].Length;

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (grid[i][j] != '1') 
                {   
                    continue;
                }
                
                ans++;
                dfs(grid, i, j);
            }
        }

        return ans;
    }
}
```

**Key Insight:** Each DFS call marks all cells in one island. Count how many times we start a new DFS to find the number of distinct islands.

### Problem: Clone Graph

Create a deep copy of a graph.

**Implementation:**

```csharp
public class Node
{
    public int val;
    public IList<Node> neighbors;

    public Node()
    {
        val = 0;
        neighbors = new List<Node>();
    }

    public Node(int _val)
    {
        val = _val;
        neighbors = new List<Node>();
    }

    public Node(int _val, List<Node> _neighbors)
    {
        val = _val;
        neighbors = _neighbors;
    }
}

public class Solution
{
    private Dictionary<Node, Node> m = new Dictionary<Node, Node>();

    public Node CloneGraph(Node node)
    {
        if (node == null)
            return null;

        if (m.ContainsKey(node))
            return m[node];

        Node cpy = new Node(node.val);
        m[node] = cpy;

        foreach (var neighbor in node.neighbors)
        {
            cpy.neighbors.Add(CloneGraph(neighbor));
        }

        return cpy;
    }
}
```

**Key Insight:** Use DFS with a dictionary to track cloned nodes. Clone each node once, then recursively clone its neighbors.

### Problem: Pacific Atlantic Water Flow

Find cells where water can flow to both Pacific and Atlantic oceans.

**Implementation:**

```csharp
public class Solution
{
    private int[][] dirs = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 } };
    private int m, n;

    public IList<IList<int>> PacificAtlantic(int[][] heights)
    {
        m = heights.Length;
        n = heights[0].Length;

        bool[][] pacific = new bool[m][];
        bool[][] atlantic = new bool[m][];

        for (int i = 0; i < m; i++)
        {
            pacific[i] = new bool[n];
            atlantic[i] = new bool[n];
        }

        // DFS from Pacific border (top and left)
        for (int i = 0; i < m; i++)
            DFS(heights, pacific, i, 0);
        for (int j = 0; j < n; j++)
            DFS(heights, pacific, 0, j);

        // DFS from Atlantic border (bottom and right)
        for (int i = 0; i < m; i++)
            DFS(heights, atlantic, i, n - 1);
        for (int j = 0; j < n; j++)
            DFS(heights, atlantic, m - 1, j);

        // Find cells reachable from both oceans
        List<IList<int>> result = new List<IList<int>>();
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (pacific[i][j] && atlantic[i][j])
                {
                    result.Add(new List<int> { i, j });
                }
            }
        }

        return result;
    }

    private void DFS(int[][] heights, bool[][] visited, int x, int y)
    {
        visited[x][y] = true;

        foreach (var dir in dirs)
        {
            int nx = x + dir[0];
            int ny = y + dir[1];

            if (nx >= 0 && nx < m && ny >= 0 && ny < n && 
                !visited[nx][ny] && heights[nx][ny] >= heights[x][y])
            {
                DFS(heights, visited, nx, ny);
            }
        }
    }
}
```

**Key Insight:** Instead of checking each cell if it reaches both oceans, do reverse DFS from ocean borders inward. A cell that both DFS traversals reach can flow to both oceans.

## Strategy 2: Breadth-First Search (BFS)

**Breadth-First Search (BFS)** is a traversal algorithm that explores vertices layer by layer, visiting all neighbors at the current level before moving to the next level. BFS uses a queue data structure and is the optimal choice for finding shortest paths in unweighted graphs, level-order traversal, and exploring graphs by distance from a source. It guarantees finding the shortest path first.

### When to use it

- Find shortest path in unweighted graph
- Level-order traversal
- Find minimum steps/distance
- Check if graph is bipartite
- Find all nodes at distance k

### Time Complexity: O(V + E) | Space Complexity: O(V)

### Problem: Word Ladder

Find the shortest transformation sequence from `beginWord` to `endWord` using a dictionary.

**Implementation:**

```csharp
public class Solution
{
    public int LadderLength(string beginWord, string endWord, IList<string> wordList)
    {
        var wordSet = new HashSet<string>(wordList);
        if (!wordSet.Contains(endWord))
            return 0;

        var queue = new Queue<(string word, int level)>();
        queue.Enqueue((beginWord, 1));

        while (queue.Count > 0)
        {
            var (currentWord, level) = queue.Dequeue();

            if (currentWord == endWord)
                return level;

            // Try changing each character
            char[] wordArray = currentWord.ToCharArray();
            for (int i = 0; i < wordArray.Length; i++)
            {
                char originalChar = wordArray[i];

                for (char c = 'a'; c <= 'z'; c++)
                {
                    if (c == originalChar) continue;

                    wordArray[i] = c;
                    string newWord = new string(wordArray);

                    if (wordSet.Contains(newWord))
                    {
                        queue.Enqueue((newWord, level + 1));
                        wordSet.Remove(newWord); // Mark as visited
                    }
                }

                wordArray[i] = originalChar; // Restore
            }
        }

        return 0;
    }
}
```

**Key Insight:** BFS explores all words at distance d before exploring words at distance d+1, guaranteeing the shortest path. Each word is visited once.

### Problem: Rotting Oranges

Find minimum time for all oranges to rot (fresh oranges adjacent to rotten ones rot each minute).

**Implementation:**

```csharp
public class Solution
{
    public int OrangesRotting(int[][] grid)
    {
        int m = grid.Length;
        int n = grid[0].Length;
        int fresh = 0;
        var queue = new Queue<(int x, int y)>();

        // Count fresh oranges and add rotten ones to queue
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (grid[i][j] == 1)
                    fresh++;
                else if (grid[i][j] == 2)
                    queue.Enqueue((i, j));
            }
        }

        if (fresh == 0) return 0;

        int[][] dirs = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 } };
        int minutes = 0;

        while (queue.Count > 0)
        {
            int size = queue.Count;
            bool rotted = false;

            for (int i = 0; i < size; i++)
            {
                var (x, y) = queue.Dequeue();

                foreach (var dir in dirs)
                {
                    int nx = x + dir[0];
                    int ny = y + dir[1];

                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1)
                    {
                        grid[nx][ny] = 2;
                        fresh--;
                        queue.Enqueue((nx, ny));
                        rotted = true;
                    }
                }
            }

            if (rotted) minutes++;
        }

        return fresh == 0 ? minutes : -1;
    }
}
```

**Key Insight:** Multi-source BFS - start from all rotten oranges simultaneously. Each BFS level represents one minute passing.

## Strategy 3: Union Find (Disjoint Set)

**Union Find** (also called Disjoint Set Union or DSU) is a data structure that tracks elements partitioned into disjoint sets and efficiently supports two operations: `find` (determine which set an element belongs to) and `union` (merge two sets). With path compression and union by rank optimizations, both operations run in nearly O(1) amortized time. Union Find excels at detecting connectivity, finding connected components, and detecting cycles in undirected graphs.

### When to use it

- Detect connected components
- Determine if adding edge creates cycle
- Find if two nodes are connected
- Kruskal's minimum spanning tree
- Dynamic connectivity problems
- Account merging, friend circles

### Time Complexity: O(α(n)) ≈ O(1) | Space Complexity: O(n)

### Problem: Graph Valid Tree

Determine if an undirected graph is a valid tree.

**Implementation:**

```csharp
public class Solution
{
    private int[] parent;

    public bool ValidTree(int n, int[][] edges)
    {
        // Tree must have exactly n-1 edges
        if (edges.Length != n - 1)
            return false;

        parent = new int[n];
        for (int i = 0; i < n; i++)
            parent[i] = i;

        // Use Union-Find to detect cycles
        foreach (var edge in edges)
        {
            int x = Find(edge[0]);
            int y = Find(edge[1]);

            if (x == y) // Cycle detected
                return false;

            parent[x] = y; // Union
        }

        return true;
    }

    private int Find(int x)
    {
        if (parent[x] != x)
            parent[x] = Find(parent[x]); // Path compression
        return parent[x];
    }
}
```

**Key Insight:** A valid tree has n-1 edges and no cycles. Use Union Find to detect cycles: if two nodes being connected already share the same root, adding an edge would create a cycle.

### Problem: Number of Connected Components in an Undirected Graph

Find the number of connected components.

**Implementation:**

```csharp
public class Solution
{
    private int[] parent;

    public int CountComponents(int n, int[][] edges)
    {
        parent = new int[n];
        for (int i = 0; i < n; i++)
            parent[i] = i;

        foreach (var edge in edges)
        {
            Union(edge[0], edge[1]);
        }

        // Count distinct roots
        HashSet<int> roots = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            roots.Add(Find(i));
        }

        return roots.Count;
    }

    private int Find(int x)
    {
        if (parent[x] != x)
            parent[x] = Find(parent[x]);
        return parent[x];
    }

    private void Union(int x, int y)
    {
        int rootX = Find(x);
        int rootY = Find(y);
        
        if (rootX != rootY)
            parent[rootX] = rootY;
    }
}
```

**Key Insight:** After processing all edges with Union Find, count the number of distinct roots (parents that point to themselves).

### Problem: Redundant Connection

Find an edge that can be removed to make the graph a tree.

**Implementation:**

```csharp
public class Solution
{
    private int[] parent;

    public int[] FindRedundantConnection(int[][] edges)
    {
        int n = edges.Length;
        parent = new int[n + 1];
        
        for (int i = 1; i <= n; i++)
            parent[i] = i;

        foreach (var edge in edges)
        {
            int x = Find(edge[0]);
            int y = Find(edge[1]);

            if (x == y) // This edge creates a cycle
                return edge;

            parent[x] = y; // Union
        }

        return new int[0];
    }

    private int Find(int x)
    {
        if (parent[x] != x)
            parent[x] = Find(parent[x]);
        return parent[x];
    }
}
```

**Key Insight:** Process edges in order. The first edge that connects two already-connected nodes is the redundant edge that creates a cycle.

## Strategy 4: Topological Sort

**Topological Sort** produces a linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge (u, v), vertex u comes before v in the ordering. This is essential for scheduling problems, build systems, course prerequisites, and task dependencies. Two common approaches exist: DFS-based (using finishing times) and BFS-based (Kahn's algorithm using indegrees). If a topological sort is impossible, the graph contains a cycle.

### When to use it

- Course prerequisites / task scheduling
- Build order problems
- Detect cycles in directed graphs
- Order tasks with dependencies
- Compile order of modules

### Time Complexity: O(V + E) | Space Complexity: O(V)

### Problem: Course Schedule

Determine if you can finish all courses given prerequisites.

**Implementation:**

```csharp
public class Solution
{
    public bool CanFinish(int numCourses, int[][] prerequisites)
    {
        List<List<int>> graph = new List<List<int>>(numCourses);

        for (int i = 0; i < numCourses; i++)
        {
            graph.Add(new List<int>());
        }

        int[] indegree = new int[numCourses];

        foreach (var p in prerequisites)
        {
            graph[p[1]].Add(p[0]);
            indegree[p[0]]++;
        }

        Queue<int> q = new Queue<int>(); // Queue of nodes with indegree = 0
        
        for (int i = 0; i < numCourses; i++)
        {
            if (indegree[i] == 0) q.Enqueue(i);
        }

        while (q.Count > 0)
        {
            int u = q.Dequeue();
            numCourses--;

            foreach (int v in graph[u])
            {
                if (--indegree[v] == 0) q.Enqueue(v);
            }
        }

        return numCourses == 0;
    }
}
```

**Key Insight:** Use BFS-based topological sort (Kahn's algorithm). Start with courses that have no prerequisites (indegree 0). If we can process all courses, no cycle exists.

### Problem: Course Schedule II

Return the ordering of courses to finish all courses.

**Implementation:**

```csharp
public class Solution
{
    public int[] FindOrder(int numCourses, int[][] prerequisites)
    {
        List<List<int>> graph = new List<List<int>>(numCourses);
        for (int i = 0; i < numCourses; i++)
            graph.Add(new List<int>());

        int[] indegree = new int[numCourses];

        foreach (var p in prerequisites)
        {
            graph[p[1]].Add(p[0]);
            indegree[p[0]]++;
        }

        Queue<int> queue = new Queue<int>();
        for (int i = 0; i < numCourses; i++)
        {
            if (indegree[i] == 0)
                queue.Enqueue(i);
        }

        List<int> order = new List<int>();

        while (queue.Count > 0)
        {
            int course = queue.Dequeue();
            order.Add(course);

            foreach (int next in graph[course])
            {
                if (--indegree[next] == 0)
                    queue.Enqueue(next);
            }
        }

        return order.Count == numCourses ? order.ToArray() : new int[0];
    }
}
```

**Key Insight:** Same as Course Schedule, but record the order in which we process courses. This order is a valid topological sort.

### Problem: Alien Dictionary

Determine the order of characters in an alien language from a sorted dictionary.

**Implementation:**

```csharp
public class Solution
{
    public string AlienOrder(string[] words)
    {
        // Build graph
        var graph = new Dictionary<char, HashSet<char>>();
        var indegree = new Dictionary<char, int>();

        // Initialize all characters
        foreach (var word in words)
        {
            foreach (var c in word)
            {
                if (!graph.ContainsKey(c))
                {
                    graph[c] = new HashSet<char>();
                    indegree[c] = 0;
                }
            }
        }

        // Build edges from adjacent words
        for (int i = 0; i < words.Length - 1; i++)
        {
            string word1 = words[i];
            string word2 = words[i + 1];
            int minLen = Math.Min(word1.Length, word2.Length);

            // Check for invalid case: word1 is prefix of word2 but comes after
            if (word1.Length > word2.Length && word1.StartsWith(word2))
                return "";

            // Find first different character
            for (int j = 0; j < minLen; j++)
            {
                if (word1[j] != word2[j])
                {
                    if (!graph[word1[j]].Contains(word2[j]))
                    {
                        graph[word1[j]].Add(word2[j]);
                        indegree[word2[j]]++;
                    }
                    break;
                }
            }
        }

        // Topological sort using BFS
        var queue = new Queue<char>();
        foreach (var kvp in indegree)
        {
            if (kvp.Value == 0)
                queue.Enqueue(kvp.Key);
        }

        StringBuilder result = new StringBuilder();

        while (queue.Count > 0)
        {
            char c = queue.Dequeue();
            result.Append(c);

            foreach (char next in graph[c])
            {
                indegree[next]--;
                if (indegree[next] == 0)
                    queue.Enqueue(next);
            }
        }

        return result.Length == graph.Count ? result.ToString() : "";
    }
}
```

**Key Insight:** Compare adjacent words to find character ordering constraints. Build a directed graph where edge c1 → c2 means c1 comes before c2. Perform topological sort.

## Strategy 5: Shortest Path Algorithms

**Shortest Path Algorithms** find the minimum-weight path between vertices. **Dijkstra's Algorithm** is the most common, using a priority queue to greedily select the nearest unvisited vertex. It works for graphs with non-negative edge weights. For unweighted graphs, BFS suffices. For negative weights (without negative cycles), use Bellman-Ford. For all-pairs shortest paths, use Floyd-Warshall.

### When to use it

- Find shortest path with weighted edges
- Network routing problems
- GPS navigation
- Minimum cost problems
- Cheapest flights

### Time Complexity: O((V + E) log V) with priority queue | Space Complexity: O(V)

### Problem: Network Delay Time

Find time for all nodes to receive a signal sent from node k.

**Implementation:**

```csharp
public class Solution
{
    public int NetworkDelayTime(int[][] times, int n, int k)
    {
        // Build adjacency list
        var graph = new Dictionary<int, List<(int node, int time)>>();
        for (int i = 1; i <= n; i++)
            graph[i] = new List<(int, int)>();

        foreach (var time in times)
        {
            graph[time[0]].Add((time[1], time[2]));
        }

        // Dijkstra's algorithm
        var dist = new int[n + 1];
        Array.Fill(dist, int.MaxValue);
        dist[k] = 0;

        var pq = new SortedSet<(int distance, int node)>();
        pq.Add((0, k));

        while (pq.Count > 0)
        {
            var (d, u) = pq.Min;
            pq.Remove(pq.Min);

            if (d > dist[u]) continue;

            foreach (var (v, time) in graph[u])
            {
                int newDist = dist[u] + time;
                if (newDist < dist[v])
                {
                    if (dist[v] != int.MaxValue)
                        pq.Remove((dist[v], v));
                    
                    dist[v] = newDist;
                    pq.Add((newDist, v));
                }
            }
        }

        int maxTime = 0;
        for (int i = 1; i <= n; i++)
        {
            if (dist[i] == int.MaxValue)
                return -1;
            maxTime = Math.Max(maxTime, dist[i]);
        }

        return maxTime;
    }
}
```

**Key Insight:** Use Dijkstra's algorithm to find shortest paths from source k to all nodes. Return the maximum distance (time for last node to receive signal).

### Problem: Cheapest Flights Within K Stops

Find the cheapest price from source to destination with at most k stops.

**Implementation:**

```csharp
public class Solution
{
    public int FindCheapestPrice(int n, int[][] flights, int src, int dst, int k)
    {
        // Build graph
        var graph = new Dictionary<int, List<(int dest, int price)>>();
        for (int i = 0; i < n; i++)
            graph[i] = new List<(int, int)>();

        foreach (var flight in flights)
        {
            graph[flight[0]].Add((flight[1], flight[2]));
        }

        // BFS with price tracking (modified Dijkstra)
        var queue = new Queue<(int city, int price, int stops)>();
        queue.Enqueue((src, 0, 0));

        int[] minPrice = new int[n];
        Array.Fill(minPrice, int.MaxValue);

        while (queue.Count > 0)
        {
            var (city, price, stops) = queue.Dequeue();

            if (stops > k + 1) continue;
            if (price >= minPrice[city]) continue;

            minPrice[city] = price;

            if (city == dst) continue;

            foreach (var (nextCity, flightPrice) in graph[city])
            {
                queue.Enqueue((nextCity, price + flightPrice, stops + 1));
            }
        }

        return minPrice[dst] == int.MaxValue ? -1 : minPrice[dst];
    }
}
```

**Key Insight:** Use BFS with constraints on stops. Track minimum price to reach each city. Stop exploring paths that exceed k stops or have higher cost than previously found paths.

## Strategy 6: Cycle Detection

**Cycle Detection** determines if a graph contains a cycle. For **undirected graphs**, use DFS and check if we revisit a node that isn't the immediate parent, or use Union Find to detect when connecting two already-connected nodes. For **directed graphs**, use DFS with three states (unvisited, visiting, visited) to detect back edges, or check if topological sort fails (Kahn's algorithm produces fewer nodes than total).

### When to use it

- Validate graph is a tree
- Detect deadlocks
- Find circular dependencies
- Check if topological sort is possible
- Detect infinite loops

### Time Complexity: O(V + E) | Space Complexity: O(V)

### Problem: Detect Cycle in Undirected Graph (using DFS)

**Implementation:**

```csharp
public class Solution
{
    public bool HasCycle(int n, int[][] edges)
    {
        var graph = new List<List<int>>(n);
        for (int i = 0; i < n; i++)
            graph.Add(new List<int>());

        foreach (var edge in edges)
        {
            graph[edge[0]].Add(edge[1]);
            graph[edge[1]].Add(edge[0]);
        }

        bool[] visited = new bool[n];

        for (int i = 0; i < n; i++)
        {
            if (!visited[i] && DFS(graph, visited, i, -1))
                return true;
        }

        return false;
    }

    private bool DFS(List<List<int>> graph, bool[] visited, int node, int parent)
    {
        visited[node] = true;

        foreach (int neighbor in graph[node])
        {
            if (!visited[neighbor])
            {
                if (DFS(graph, visited, neighbor, node))
                    return true;
            }
            else if (neighbor != parent)
            {
                return true; // Cycle detected
            }
        }

        return false;
    }
}
```

**Key Insight:** In undirected graph, a cycle exists if we visit a node that's already visited and it's not our parent (the node we came from).

### Problem: Detect Cycle in Directed Graph

**Implementation:**

```csharp
public class Solution
{
    private enum State { Unvisited, Visiting, Visited }

    public bool HasCycle(int n, int[][] edges)
    {
        var graph = new List<List<int>>(n);
        for (int i = 0; i < n; i++)
            graph.Add(new List<int>());

        foreach (var edge in edges)
        {
            graph[edge[0]].Add(edge[1]);
        }

        State[] states = new State[n];

        for (int i = 0; i < n; i++)
        {
            if (states[i] == State.Unvisited && DFS(graph, states, i))
                return true;
        }

        return false;
    }

    private bool DFS(List<List<int>> graph, State[] states, int node)
    {
        states[node] = State.Visiting;

        foreach (int neighbor in graph[node])
        {
            if (states[neighbor] == State.Visiting)
                return true; // Back edge - cycle detected

            if (states[neighbor] == State.Unvisited && DFS(graph, states, neighbor))
                return true;
        }

        states[node] = State.Visited;
        return false;
    }
}
```

**Key Insight:** Use three states. If we encounter a node in "Visiting" state, we've found a back edge (cycle). Visiting → Visited transition means subtree is cycle-free.

## Strategy 7: Graph Coloring / Bipartite

**Graph Coloring** assigns colors to vertices such that no two adjacent vertices share the same color. **Bipartite** graphs can be 2-colored (two sets where edges only connect between sets). To check bipartiteness, use BFS or DFS to color nodes alternately (0 and 1). If we encounter a conflict (adjacent nodes with same color), the graph is not bipartite. This is useful for matching problems, scheduling with conflicts, and detecting odd-length cycles.

### When to use it

- Check if graph is bipartite
- Two-coloring problems
- Matching problems (jobs to workers)
- Detect odd-length cycles
- Partition into two independent sets

### Time Complexity: O(V + E) | Space Complexity: O(V)

### Problem: Is Graph Bipartite?

**Implementation:**

```csharp
public class Solution
{
    public bool IsBipartite(int[][] graph)
    {
        int n = graph.Length;
        int[] colors = new int[n];
        Array.Fill(colors, -1);

        for (int i = 0; i < n; i++)
        {
            if (colors[i] == -1)
            {
                if (!BFS(graph, colors, i))
                    return false;
            }
        }

        return true;
    }

    private bool BFS(int[][] graph, int[] colors, int start)
    {
        var queue = new Queue<int>();
        queue.Enqueue(start);
        colors[start] = 0;

        while (queue.Count > 0)
        {
            int node = queue.Dequeue();

            foreach (int neighbor in graph[node])
            {
                if (colors[neighbor] == -1)
                {
                    colors[neighbor] = 1 - colors[node];
                    queue.Enqueue(neighbor);
                }
                else if (colors[neighbor] == colors[node])
                {
                    return false; // Conflict - not bipartite
                }
            }
        }

        return true;
    }
}
```

**Key Insight:** Try to 2-color the graph using BFS. Assign opposite colors to neighbors. If two adjacent nodes end up with the same color, the graph is not bipartite.

## Strategy 8: Brute Force

**Brute Force** approaches for graphs typically involve exploring all possible paths, trying all combinations, or using simpler but less efficient algorithms. While these solutions may have exponential time complexity or higher polynomial complexity, they're straightforward to implement and useful for small graphs or as a baseline before optimization.

### When to use it

- Graph size is very small (< 10 nodes)
- Need to explore all paths
- Establishing correctness baseline
- Prototype before optimization

### Time Complexity: O(V!) to O(2^V) | Space Complexity: O(V)

**Example: Find All Paths (Brute Force DFS)**

```csharp
public class Solution
{
    public IList<IList<int>> AllPathsSourceTarget(int[][] graph)
    {
        List<IList<int>> result = new List<IList<int>>();
        List<int> path = new List<int> { 0 };
        DFS(graph, 0, path, result);
        return result;
    }

    private void DFS(int[][] graph, int node, List<int> path, List<IList<int>> result)
    {
        if (node == graph.Length - 1)
        {
            result.Add(new List<int>(path));
            return;
        }

        foreach (int neighbor in graph[node])
        {
            path.Add(neighbor);
            DFS(graph, neighbor, path, result);
            path.RemoveAt(path.Count - 1); // Backtrack
        }
    }
}
```

**Key Insight:** Use DFS with backtracking to explore all possible paths. This works for small graphs but has exponential time complexity.

## Strategy Selection Guide

| Problem Type | Strategy | Time | Space |
|-------------|----------|------|-------|
| Find connected components | DFS or Union Find | O(V+E) | O(V) |
| Shortest path (unweighted) | BFS | O(V+E) | O(V) |
| Shortest path (weighted) | Dijkstra | O((V+E)log V) | O(V) |
| Detect cycles (undirected) | DFS or Union Find | O(V+E) | O(V) |
| Detect cycles (directed) | DFS (3 states) | O(V+E) | O(V) |
| Task scheduling | Topological Sort | O(V+E) | O(V) |
| Check if bipartite | BFS/DFS 2-coloring | O(V+E) | O(V) |
| Dynamic connectivity | Union Find | O(α(n)) | O(n) |

## Pattern Recognition Tips

1. **"Shortest path"?** → BFS (unweighted) or Dijkstra (weighted)
2. **"Connected components"?** → DFS or Union Find
3. **"Cycle detection"?** → DFS (directed) or Union Find (undirected)
4. **"Prerequisites/dependencies"?** → Topological Sort
5. **"Minimum cost/time to reach all"?** → Dijkstra or BFS
6. **"Two groups with no edges within"?** → Bipartite check
7. **"All possible paths"?** → DFS with backtracking (brute force)

## Common Graph Representations

### Adjacency List (Most Common)

```csharp
// Using List<List<int>>
var graph = new List<List<int>>(n);
for (int i = 0; i < n; i++)
    graph.Add(new List<int>());

graph[u].Add(v); // Add edge u → v
```

### Adjacency Matrix

```csharp
// Using 2D array
int[][] graph = new int[n][];
for (int i = 0; i < n; i++)
    graph[i] = new int[n];

graph[u][v] = 1; // Add edge u → v
```

### Edge List

```csharp
// For Union Find or Kruskal's
int[][] edges = new int[][] { new int[] {u, v, weight}, ... };
```

## Common Pitfalls

1. **Forgetting to mark visited**: Leads to infinite loops in cyclic graphs
2. **Wrong graph representation**: Choose based on problem (sparse vs dense)
3. **Directed vs undirected**: Add both u→v and v→u for undirected
4. **Not handling disconnected components**: Iterate through all nodes as potential starts
5. **Integer overflow**: Use `long` for distance calculations
6. **Not checking edge cases**: Empty graph, single node, disconnected graph

## Conclusion

Mastering graph algorithms is essential for solving complex relationship and connectivity problems. The key is to:

1. **Identify the graph structure** - Is it directed? Weighted? Dense or sparse?
2. **Choose the right algorithm** - DFS for exploration, BFS for shortest paths, Union Find for connectivity
3. **Pick optimal representation** - Adjacency list for sparse, matrix for dense
4. **Handle edge cases** - Disconnected components, cycles, single nodes
5. **Optimize space and time** - Use visited sets, early termination, efficient data structures

Practice these strategies on LeetCode to build intuition for graph problems. With experience, you'll recognize graph patterns instantly and select the optimal approach efficiently! 🚀

## References

_LeetCode. (2024). Graph Problems. https://leetcode.com/tag/graph/_

_Cormen, T. H., Leiserson, C. E., Rivert, R. L. & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press._

_Dijkstra, E. W. (1959). A Note on Two Problems in Connexion with Graphs. Numerische Mathematik._

_Tarjan, R. E. (1975). Efficiency of a Good But Not Linear Set Union Algorithm. Journal of the ACM._
