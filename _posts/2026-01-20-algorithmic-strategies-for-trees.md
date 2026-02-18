---
title: Algorithmic Strategies for Trees 🌲
description: Master the most common algorithmic strategies to solve tree problems in LeetCode. Learn how to identify patterns and apply the right approach with real C# implementations.
Author: Jonas Lara
date: 2026-01-20 00:00:00 +0000
categories: [Algorithms, Data Structures, LeetCode]
tags: [trees, binary trees, dfs, bfs, trie, recursion, inorder, preorder, postorder, traversal]
image:
  path: /assets/img/post/algorithmic-strategies-for-trees/trees-kp.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/algorithmic-strategies-for-trees/trees-kp.png
  alt: Binary Tree Visualization
---

## Introduction

Trees are hierarchical data structures that consist of nodes connected by edges, with a single root node and no cycles. **The key to solving tree problems efficiently is recognizing patterns and applying the right algorithmic strategy**. This guide explores the most common strategies with real C# implementations from LeetCode problems.

## Common Algorithmic Strategies

When approaching tree problems, you'll typically encounter one of these strategies:

- **Depth-First Search (DFS)**: Explore as deep as possible before backtracking
- **Breadth-First Search (BFS)**: Explore level by level using a queue
- **Recursion**: Leverage the recursive nature of trees
- **Tree Traversals**: Inorder, Preorder, Postorder patterns
- **Trie (Prefix Tree)**: Specialized tree for string operations
- **Binary Search Tree Properties**: Leverage BST ordering
- **Divide and Conquer**: Split problem into left and right subtrees
- **Brute Force**: Try all possible approaches

## Strategy 1: Depth-First Search (DFS)

**Depth-First Search (DFS)** is a traversal algorithm that explores as far as possible along each branch before backtracking. For trees, DFS can be implemented recursively (using the call stack) or iteratively (using an explicit stack). DFS is the natural choice for most tree problems because of the recursive structure of trees. It's particularly effective when you need to explore all paths, find depths, or compute values that depend on subtree results.

### When to use it

- Need to explore all nodes
- Calculate depth or height
- Find paths from root to leaf
- Problems requiring backtracking
- Validate tree properties

### Time Complexity: O(n) | Space Complexity: O(h) where h is height

### Problem: Maximum Depth of Binary Tree

Find the maximum depth (height) of a binary tree.

**Implementation:**

```csharp
public class TreeNode
{
    public int val;
    public TreeNode left;
    public TreeNode right;

    public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
    {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

public class Solution
{
    public int MaxDepth(TreeNode root)
    {
        return root != null ? 1 + Math.Max(MaxDepth(root.left), MaxDepth(root.right)) : 0;
    }
}
```

**Key Insight:** The depth of a tree is 1 plus the maximum depth of its subtrees. Base case: empty tree has depth 0.

### Problem: Invert Binary Tree

Invert a binary tree (swap left and right children at every node).

**Implementation:**

```csharp
public class Solution
{
    public TreeNode InvertTree(TreeNode root)
    {
        if (root == null)
        {
            return null;
        }

        Swap(ref root.left, ref root.right);
        InvertTree(root.left);
        InvertTree(root.right);

        return root;
    }

    private void Swap<T>(ref T x, ref T y)
    {
        T temp = x;
        x = y;
        y = temp;
    }
}
```

**Key Insight:** Recursively swap children at each node. Process current node first (preorder), then recurse on children.

### Problem: Validate Binary Search Tree

Determine if a binary tree is a valid BST.

**Implementation:**

```csharp
public class Solution
{
    public bool IsValidBST(TreeNode root)
    {
        return Validate(root, long.MinValue, long.MaxValue);
    }

    private bool Validate(TreeNode node, long min, long max)
    {
        if (node == null)
            return true;

        if (node.val <= min || node.val >= max)
            return false;

        return Validate(node.left, min, node.val) && 
               Validate(node.right, node.val, max);
    }
}
```

**Key Insight:** Pass valid range constraints down the tree. Left subtree must be < node.val, right subtree must be > node.val.

## Strategy 2: Breadth-First Search (BFS)

**Breadth-First Search (BFS)** is a traversal algorithm that explores nodes level by level, starting from the root. It uses a queue data structure to track nodes at the current level before moving to the next level. BFS is the optimal choice when you need level-order traversal, find shortest paths in unweighted trees, or process nodes by their distance from the root. Unlike DFS which uses recursion naturally, BFS is typically implemented iteratively with a queue.

### When to use it

- Level-order traversal needed
- Find shortest path/minimum depth
- Process nodes by levels
- Need to explore layer by layer
- Find nodes at specific distance

### Time Complexity: O(n) | Space Complexity: O(w) where w is max width

### Problem: Binary Tree Level Order Traversal

Return level-order traversal of a binary tree (values level by level).

**Implementation:**

```csharp
public class Solution
{
    public IList<IList<int>> LevelOrder(TreeNode root)
    {
        if (root == null)
            return new List<IList<int>>();

        List<IList<int>> ans = new List<IList<int>>();
        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            int levelSize = queue.Count;
            List<int> currentLevel = new List<int>();

            for (int i = 0; i < levelSize; i++)
            {
                var node = queue.Dequeue();
                currentLevel.Add(node.val);

                if (node.left != null)
                    queue.Enqueue(node.left);

                if (node.right != null)
                    queue.Enqueue(node.right);
            }

            ans.Add(currentLevel);
        }

        return ans;
    }
}
```

**Key Insight:** Process all nodes at current level before moving to next level. Track level size to know when to move to next level.

### Problem: Binary Tree Right Side View

Return values of nodes you can see when looking at the tree from the right side.

**Implementation:**

```csharp
public class Solution
{
    public IList<int> RightSideView(TreeNode root)
    {
        List<int> result = new List<int>();
        if (root == null) return result;

        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            int levelSize = queue.Count;
            
            for (int i = 0; i < levelSize; i++)
            {
                TreeNode node = queue.Dequeue();
                
                // Add rightmost node of each level
                if (i == levelSize - 1)
                    result.Add(node.val);

                if (node.left != null)
                    queue.Enqueue(node.left);
                if (node.right != null)
                    queue.Enqueue(node.right);
            }
        }

        return result;
    }
}
```

**Key Insight:** Use BFS to traverse level by level, but only record the last node at each level (rightmost visible node).

## Strategy 3: Tree Traversals (Inorder, Preorder, Postorder)

**Tree Traversals** are systematic ways to visit all nodes in a tree. The three main traversal orders are:
- **Inorder (Left-Root-Right)**: Visits left subtree, then root, then right subtree. For BSTs, this gives sorted order.
- **Preorder (Root-Left-Right)**: Visits root first, then left, then right. Used for creating copies or prefix expressions.
- **Postorder (Left-Right-Root)**: Visits children before parent. Used for deletion or postfix expressions.

Understanding when to use each traversal is crucial for solving tree problems efficiently.

### When to use it

- **Inorder**: BST problems, sorted output, find kth smallest
- **Preorder**: Tree serialization, creating copies, prefix notation
- **Postorder**: Tree deletion, calculating subtree properties, postfix notation
- Need specific visiting order

### Time Complexity: O(n) | Space Complexity: O(h) for recursion

### Problem: Kth Smallest Element in a BST

Find the kth smallest element in a BST using inorder traversal.

**Implementation:**

```csharp
public class Solution
{
    private int count = 0;
    private int result = 0;

    public int KthSmallest(TreeNode root, int k)
    {
        Inorder(root, k);
        return result;
    }

    private void Inorder(TreeNode node, int k)
    {
        if (node == null) return;

        Inorder(node.left, k);
        
        count++;
        if (count == k)
        {
            result = node.val;
            return;
        }

        Inorder(node.right, k);
    }
}
```

**Key Insight:** Inorder traversal of BST visits nodes in ascending order. Count nodes until we reach the kth one.

### Problem: Construct Binary Tree from Preorder and Inorder Traversal

Build a tree given preorder and inorder traversal arrays.

**Implementation:**

```csharp
public class Solution
{
    private int preorderIndex = 0;
    private Dictionary<int, int> inorderIndexMap = new Dictionary<int, int>();

    public TreeNode BuildTree(int[] preorder, int[] inorder)
    {
        // Build hashmap for quick inorder index lookup
        for (int i = 0; i < inorder.Length; i++)
        {
            inorderIndexMap[inorder[i]] = i;
        }

        return Build(preorder, 0, preorder.Length - 1);
    }

    private TreeNode Build(int[] preorder, int left, int right)
    {
        if (left > right)
            return null;

        int rootVal = preorder[preorderIndex++];
        TreeNode root = new TreeNode(rootVal);

        // Build left and right subtrees
        int inorderIndex = inorderIndexMap[rootVal];
        root.left = Build(preorder, left, inorderIndex - 1);
        root.right = Build(preorder, inorderIndex + 1, right);

        return root;
    }
}
```

**Key Insight:** Preorder gives root first. Find root in inorder to determine left and right subtree boundaries.

### Problem: Binary Tree Maximum Path Sum

Find the maximum path sum between any two nodes.

**Implementation:**

```csharp
public class Solution
{
    private int maxSum = int.MinValue;

    public int MaxPathSum(TreeNode root)
    {
        MaxGain(root);
        return maxSum;
    }

    private int MaxGain(TreeNode node)
    {
        if (node == null)
            return 0;

        // Recursively get max gain from left and right (ignore negative)
        int leftGain = Math.Max(MaxGain(node.left), 0);
        int rightGain = Math.Max(MaxGain(node.right), 0);

        // Path through this node
        int pathSum = node.val + leftGain + rightGain;
        maxSum = Math.Max(maxSum, pathSum);

        // Return max gain if continuing to parent
        return node.val + Math.Max(leftGain, rightGain);
    }
}
```

**Key Insight:** Use postorder traversal (process children first). At each node, consider both the path through this node and the max gain continuing upward.

## Strategy 4: Trie (Prefix Tree)

A **Trie** (pronounced "try") is a specialized tree data structure used for efficient string operations. Each node represents a character, and paths from root to nodes spell out strings. Tries are particularly powerful for prefix-based operations, word searches, and autocomplete features. They trade space for time, using O(m) space per word but providing O(m) lookup time where m is the word length, regardless of how many words are stored.

### When to use it

- Prefix matching problems
- Autocomplete functionality
- Dictionary/word search problems
- Storing strings with common prefixes efficiently
- Need fast prefix queries

### Time Complexity: O(m) where m is word length | Space Complexity: O(n*m)

### Problem: Implement Trie (Prefix Tree)

Implement a trie with insert, search, and startsWith methods.

**Implementation:**

```csharp
public class TrieNode
{
    public TrieNode[] Next { get; } = new TrieNode[26];
    public bool Word { get; set; }
}

public class Trie
{
    private TrieNode root = new TrieNode();

    private TrieNode Find(string word)
    {
        var node = root;
        
        foreach (char c in word)
        {
            if (node.Next[c - 'a'] == null)
            {
                return null;
            }

            node = node.Next[c - 'a'];
        }

        return node;
    }

    public void Insert(string word)
    {
        var node = root;
        foreach (char c in word)
        {
            if (node.Next[c - 'a'] == null)
            {
                node.Next[c - 'a'] = new TrieNode();
            }

            node = node.Next[c - 'a'];
        }
        
        node.Word = true;
    }

    public bool Search(string word)
    {
        var node = Find(word);
        return node != null && node.Word;
    }

    public bool StartsWith(string prefix)
    {
        return Find(prefix) != null;
    }
}
```

**Key Insight:** Each TrieNode contains an array of children (26 for lowercase letters). The `Word` flag marks end of valid words.

### Problem: Design Add and Search Words Data Structure

Design a data structure that supports adding words and searching with '.' wildcard.

**Implementation:**

```csharp
public class TrieNode
{
    public TrieNode[] Children = new TrieNode[26];
    public bool IsWord = false;
}

public class WordDictionary
{
    private TrieNode root;

    public WordDictionary()
    {
        root = new TrieNode();
    }

    public void AddWord(string word)
    {
        TrieNode node = root;
        foreach (char c in word)
        {
            int index = c - 'a';
            if (node.Children[index] == null)
                node.Children[index] = new TrieNode();
            node = node.Children[index];
        }
        node.IsWord = true;
    }

    public bool Search(string word)
    {
        return SearchHelper(word, 0, root);
    }

    private bool SearchHelper(string word, int index, TrieNode node)
    {
        if (node == null)
            return false;

        if (index == word.Length)
            return node.IsWord;

        char c = word[index];

        if (c == '.')
        {
            // Try all possible children
            for (int i = 0; i < 26; i++)
            {
                if (SearchHelper(word, index + 1, node.Children[i]))
                    return true;
            }
            return false;
        }
        else
        {
            return SearchHelper(word, index + 1, node.Children[c - 'a']);
        }
    }
}
```

**Key Insight:** For wildcard '.', try all 26 possible children recursively. Use DFS to explore all possible matches.

## Strategy 5: Binary Search Tree Properties

**Binary Search Trees (BST)** have a special property: for every node, all values in the left subtree are smaller, and all values in the right subtree are larger. This ordering property enables efficient search, insertion, and deletion operations in O(log n) average time for balanced trees. Many BST problems leverage this property to find elements, validate structure, or find successors/predecessors efficiently.

### When to use it

- Tree is explicitly a BST
- Need to find/insert/delete in O(log n)
- Find kth smallest/largest element
- Range queries
- Find successor/predecessor

### Time Complexity: O(log n) to O(n) | Space Complexity: O(h)

### Problem: Lowest Common Ancestor of a Binary Search Tree

Find the LCA of two nodes in a BST.

**Implementation:**

```csharp
public class Solution
{
    public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q)
    {
        // Both nodes in left subtree
        if (p.val < root.val && q.val < root.val)
            return LowestCommonAncestor(root.left, p, q);
        
        // Both nodes in right subtree
        if (p.val > root.val && q.val > root.val)
            return LowestCommonAncestor(root.right, p, q);
        
        // Split point (one left, one right) or one equals root
        return root;
    }
}
```

**Key Insight:** Use BST property - if both nodes are smaller, LCA is in left subtree; if both larger, in right subtree; otherwise current node is LCA.

## Strategy 6: Divide and Conquer

**Divide and Conquer** is a powerful paradigm for tree problems where you split the problem into independent subproblems for left and right subtrees, solve them recursively, and combine the results. This approach works naturally with the binary structure of trees. Many tree problems like finding diameter, balanced checking, or computing subtree properties use this pattern.

### When to use it

- Problem can be split into independent subproblems
- Need results from both subtrees
- Combining subtree results is straightforward
- Tree structure naturally suggests division

### Time Complexity: O(n) | Space Complexity: O(h)

### Problem: Same Tree

Check if two trees are identical.

**Implementation:**

```csharp
public class Solution
{
    public bool IsSameTree(TreeNode p, TreeNode q)
    {
        if (p == null && q == null)
            return true;
        
        if (p == null || q == null)
            return false;
        
        if (p.val != q.val)
            return false;

        return IsSameTree(p.left, q.left) && IsSameTree(p.right, q.right);
    }
}
```

**Key Insight:** Trees are identical if roots match and both left and right subtrees are identical.

### Problem: Subtree of Another Tree

Check if tree s is a subtree of tree t.

**Implementation:**

```csharp
public class Solution
{
    public bool IsSubtree(TreeNode root, TreeNode subRoot)
    {
        if (root == null)
            return false;

        if (IsSameTree(root, subRoot))
            return true;

        return IsSubtree(root.left, subRoot) || IsSubtree(root.right, subRoot);
    }

    private bool IsSameTree(TreeNode p, TreeNode q)
    {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        if (p.val != q.val) return false;

        return IsSameTree(p.left, q.left) && IsSameTree(p.right, q.right);
    }
}
```

**Key Insight:** Check if trees match at current root, otherwise recursively check left and right subtrees.

## Strategy 7: Serialization and Deserialization

**Serialization** converts a tree structure into a string representation, while **deserialization** reconstructs the tree from the string. This is useful for storing trees, network transmission, or caching. The key challenge is encoding tree structure (including null nodes) in a way that allows unambiguous reconstruction.

### When to use it

- Need to save/load tree structures
- Network transmission
- Caching tree data
- Clone or copy trees

### Time Complexity: O(n) | Space Complexity: O(n)

### Problem: Serialize and Deserialize Binary Tree

**Implementation:**

```csharp
public class Codec
{
    // Encodes a tree to a single string (preorder)
    public string serialize(TreeNode root)
    {
        if (root == null)
            return "null";

        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }

    // Decodes your encoded data to tree
    public TreeNode deserialize(string data)
    {
        Queue<string> nodes = new Queue<string>(data.Split(','));
        return Deserialize(nodes);
    }

    private TreeNode Deserialize(Queue<string> nodes)
    {
        string val = nodes.Dequeue();
        
        if (val == "null")
            return null;

        TreeNode root = new TreeNode(int.Parse(val));
        root.left = Deserialize(nodes);
        root.right = Deserialize(nodes);
        
        return root;
    }
}
```

**Key Insight:** Use preorder traversal for serialization. Include "null" markers to preserve structure. Deserialize using same preorder pattern.

## Strategy 8: Brute Force

**Brute Force** approaches for trees typically involve exploring all possibilities, often with nested recursion or converting the tree to another data structure (like an array) first. While these solutions may have higher time or space complexity, they're often easier to understand and implement, making them good starting points before optimization.

### When to use it

- Problem size is small
- Correctness is priority
- Establishing baseline for optimization
- Quick prototype needed

### Time Complexity: O(n²) to O(n*h) | Space Complexity: O(n)

**Example: Converting to array first**

```csharp
public class Solution
{
    public IList<int> InorderTraversalBruteForce(TreeNode root)
    {
        List<int> result = new List<int>();
        Stack<TreeNode> stack = new Stack<TreeNode>();
        List<TreeNode> visited = new List<TreeNode>();

        if (root != null)
            stack.Push(root);

        while (stack.Count > 0)
        {
            TreeNode node = stack.Peek();

            if (node.left != null && !visited.Contains(node.left))
            {
                stack.Push(node.left);
            }
            else
            {
                result.Add(node.val);
                visited.Add(node);
                stack.Pop();

                if (node.right != null)
                    stack.Push(node.right);
            }
        }

        return result;
    }
}
```

**Key Insight:** Using a visited list simplifies logic but uses extra O(n) space. Optimized version tracks state differently.

## Strategy Selection Guide

| Problem Type | Strategy | Time | Space |
|-------------|----------|------|-------|
| Maximum depth/height | DFS (Recursion) | O(n) | O(h) |
| Level-order traversal | BFS (Queue) | O(n) | O(w) |
| Kth smallest in BST | Inorder Traversal | O(n) | O(h) |
| Prefix matching | Trie | O(m) | O(n*m) |
| Validate BST | DFS with constraints | O(n) | O(h) |
| Find LCA in BST | BST Properties | O(log n) | O(1) |
| Check if same tree | Divide and Conquer | O(n) | O(h) |
| Serialize tree | Preorder Traversal | O(n) | O(n) |

## Pattern Recognition Tips

1. **Need level information?** → Think BFS
2. **Need depth information?** → Think DFS
3. **Binary Search Tree?** → Think BST properties or Inorder
4. **String/prefix problem?** → Think Trie
5. **Build tree from traversals?** → Think Preorder + Inorder or Postorder + Inorder
6. **Combine subtree results?** → Think Divide and Conquer
7. **Path from root to node?** → Think DFS with path tracking

## Common Tree Problem Patterns

### 1. Bottom-Up (Postorder)
Process children before parent, return values upward.
```csharp
int Process(TreeNode node)
{
    if (node == null) return 0;
    int left = Process(node.left);
    int right = Process(node.right);
    return Combine(left, right, node.val);
}
```

### 2. Top-Down (Preorder)
Pass values down from parent to children.
```csharp
void Process(TreeNode node, int valueFromParent)
{
    if (node == null) return;
    // Use valueFromParent
    Process(node.left, newValue);
    Process(node.right, newValue);
}
```

### 3. Level Processing (BFS)
Process all nodes at each level.
```csharp
while (queue.Count > 0)
{
    int levelSize = queue.Count;
    for (int i = 0; i < levelSize; i++)
    {
        var node = queue.Dequeue();
        // Process node
    }
}
```

## Common Pitfalls

1. **Not handling null nodes**: Always check `node != null` before accessing
2. **Confusing left/right**: Be careful with left and right child pointers
3. **Wrong traversal order**: Match traversal to problem requirements
4. **Forgetting base cases**: Empty tree, single node, leaf nodes
5. **Stack overflow**: Deep recursion on unbalanced trees
6. **Modifying tree during traversal**: Can lose references or create cycles

## Conclusion

Mastering these algorithmic strategies is essential for solving tree problems efficiently. The key is to:

1. **Identify the pattern** - Is it about depth, levels, paths, or prefixes?
2. **Choose the right traversal** - DFS for depth, BFS for levels, inorder for BST
3. **Leverage tree properties** - BST ordering, parent-child relationships
4. **Handle edge cases** - Null nodes, single nodes, unbalanced trees
5. **Optimize space** - Prefer recursion for clean code, iteration for space efficiency

Practice these strategies on LeetCode to build intuition for which approach to use. With time, pattern recognition becomes second nature, and you'll quickly identify the optimal strategy for any tree problem you encounter. 🚀

## References

_LeetCode. (2024). Tree Problems. https://leetcode.com/tag/tree/_

_LeetCode. (2024). Binary Tree Problems. https://leetcode.com/tag/binary-tree/_

_Cormen, T. H., Leiserson, C. E., Rivert, R. L. & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press._

_Fredkin, E. (1960). Trie Memory. Communications of the ACM._
