---
title: Algorithmic Strategies for Dynamic Programming 💡
description: Master dynamic programming strategies to solve optimization problems in LeetCode. Learn bottom-up, top-down, memoization, and tabulation with real C# implementations.
Author: Jonas Lara
date: 2026-01-14 00:00:00 +0000
categories: [Algorithms, Data Structures, LeetCode]
tags: [dynamic programming, dp, memoization, tabulation, optimization]
image:
  path: /assets/img/post/algorithmic-strategies-for-dp/dp.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/algorithmic-strategies-for-dp/dp.png
  alt: Morphing a lobster into a head via dynamic programming
---

## Introduction

Dynamic Programming (DP) is one of the most powerful problem-solving paradigms in computer science. **The key to DP is recognizing overlapping subproblems and optimal substructure - solving smaller problems once and reusing their solutions**. This guide explores the fundamental DP strategies with real C# implementations from LeetCode problems.

## What is Dynamic Programming?

Dynamic Programming solves complex problems by:
1. **Breaking them into simpler overlapping subproblems**
2. **Storing solutions to avoid redundant computation**
3. **Building optimal solutions from optimal subproblem solutions**

The two main approaches are:
- **Top-Down (Memoization)**: Recursion + caching results
- **Bottom-Up (Tabulation)**: Iteratively fill a table from base cases

## Common Algorithmic Strategies

When approaching DP problems, you'll typically encounter these strategies:

- **Bottom-Up (Tabulation)**: Build solution iteratively from base cases
- **Top-Down (Memoization)**: Recursion with caching
- **Space Optimization**: Reduce space from O(n) to O(1)
- **State Machines**: Track multiple states (buy/sell, rob/skip)
- **2D DP**: Problems with two dimensions (strings, grids)

## Strategy 1: Fibonacci-style Bottom-Up

**Fibonacci-style Bottom-Up** is the simplest DP pattern where each state depends only on the previous 1-2 states. This appears in problems like climbing stairs, where the number of ways to reach step n equals ways(n-1) + ways(n-2). The pattern: compute base cases, then iteratively compute each subsequent state using the recurrence relation. Space can often be optimized to O(1) by keeping only recent states.

### When to use it

- Current state depends on previous 1-2 states
- Fibonacci-like recurrence relation
- Linear sequence of decisions
- Can optimize space to O(1)

### Time Complexity: O(n) | Space Complexity: O(1)

### Problem: Climbing Stairs

You're climbing stairs with n steps. Each time you can climb 1 or 2 steps. How many distinct ways to reach the top?

**Implementation:**

```csharp
public class Solution
{
    public int ClimbStairs(int n)
    {
        int ans = 1;
        int prev = 1;
        
        while (--n > 0)
        {
            int temp = ans;
            ans += prev;
            prev = temp;
        }
        
        return ans;
    }
}
```

**Key Insight:** This is Fibonacci in disguise. Ways(n) = Ways(n-1) + Ways(n-2). From any step, you can reach it from 1-step-back or 2-steps-back. Only need to track last two values.

**Recurrence Relation:**
```
f(0) = 1 (base case: already at top)
f(1) = 1 (one way: one 1-step)
f(n) = f(n-1) + f(n-2) for n ≥ 2
```

**Example:**
```
n = 5
f(0)=1, f(1)=1
f(2) = f(1)+f(0) = 2
f(3) = f(2)+f(1) = 3
f(4) = f(3)+f(2) = 5
f(5) = f(4)+f(3) = 8
```

### Brute Force Alternative: O(2^n)

```csharp
// Recursive without memoization - exponential time
public int ClimbStairs(int n)
{
    if (n <= 1) return 1;
    return ClimbStairs(n - 1) + ClimbStairs(n - 2);
}
```

**Why Bottom-Up is Better:** O(n) vs O(2^n). Avoids recalculating same subproblems millions of times.

## Strategy 2: State Machine DP

**State Machine DP** models problems where you transition between distinct states at each step. For house robber, states are "just robbed" vs "didn't rob" - transitions follow rules (can't rob adjacent houses). Track value of each state, update based on previous states and transition rules. This pattern appears in buy/sell stock, scheduling, and resource allocation problems.

### When to use it

- Multiple distinct states at each position
- State transitions follow specific rules
- Need to track "currently in state X"
- Binary choices (rob/skip, buy/sell)

### Time Complexity: O(n) | Space Complexity: O(1)

### Problem: House Robber

Rob houses along a street (can't rob adjacent houses). Maximize money robbed.

**Implementation:**

```csharp
public class Solution
{
    public int Rob(int[] nums)
    {
        int rob = 0;   // Max if we rob current house
        int skip = 0;  // Max if we skip current house
        
        foreach (int n in nums)
        {
            int r = n + skip;  // Rob current = current value + previous skip
            int s = Math.Max(rob, skip);  // Skip current = max of previous states
            rob = r;
            skip = s;
        }
        
        return Math.Max(rob, skip);
    }
}
```

**Key Insight:** Track two states: rob or skip current house. If rob current, add to previous skip (can't rob adjacent). If skip, take max of previous rob/skip.

**State Transitions:**
```
At house i:
  rob[i] = nums[i] + skip[i-1]  (rob current + can't rob previous)
  skip[i] = max(rob[i-1], skip[i-1])  (best without current)
```

**Example:**
```
nums = [2,7,9,3,1]
Position 0: rob=2, skip=0
Position 1: rob=7(7+0), skip=2(max(2,0))
Position 2: rob=11(9+2), skip=7(max(7,2))
Position 3: rob=10(3+7), skip=11(max(11,7))
Position 4: rob=12(1+11), skip=11(max(10,11))
Result: max(12,11) = 12
```

### Brute Force Alternative: O(2^n)

```csharp
// Try all combinations
public int Rob(int[] nums)
{
    return RobRecursive(nums, 0);
}

private int RobRecursive(int[] nums, int i)
{
    if (i >= nums.Length) return 0;
    // Take current or skip current
    return Math.Max(nums[i] + RobRecursive(nums, i+2), 
                    RobRecursive(nums, i+1));
}
```

**Why State Machine is Better:** O(n) single pass vs O(2^n) trying all combinations. Tracks states efficiently.

## Strategy 3: Tabulation with Array

**Tabulation** builds a table (array) bottom-up from base cases to the target. Each entry represents the optimal solution to a subproblem. The recurrence relation defines how to fill each cell based on previous cells. This approach guarantees no stack overflow (unlike recursion) and makes dependency order explicit. Common in coin change, knapsack, and path counting problems.

### When to use it

- Need to compute all subproblem solutions
- Clear iterative dependency order
- Avoid recursion stack overflow
- Can visualize as filling a table

### Time Complexity: O(n×m) | Space Complexity: O(n) to O(n×m)

### Problem: Coin Change

Find minimum number of coins to make amount (coins array given).

**Implementation:**

```csharp
public class Solution
{
    public int CoinChange(int[] coins, int amount)
    {
        int inf = 0x3f3f3f3f;
        int[] dp = new int[amount + 1];
        Array.Fill(dp, inf);
        dp[0] = 0;  // Base case: 0 coins for amount 0
        
        for (int t = 1; t <= amount; t++)
        {
            for (int i = 0; i < coins.Length; i++)
            {
                if (t - coins[i] >= 0)
                {
                    dp[t] = Math.Min(dp[t], 1 + dp[t - coins[i]]);
                }
            }
        }
        
        return dp[amount] == inf ? -1 : dp[amount];
    }
}
```

**Key Insight:** For each amount, try all coins. If coin fits, answer is 1 + dp[amount - coin]. Take minimum across all coins. Build from 0 to target amount.

**Recurrence:**
```
dp[0] = 0
dp[amount] = min(dp[amount - coin] + 1) for all coins where coin ≤ amount
```

**Example:**
```
coins = [1,2,5], amount = 11
dp[0] = 0
dp[1] = 1 (one 1-coin)
dp[2] = 1 (one 2-coin)
dp[3] = 2 (1+2 or 1+1+1, min=2)
...
dp[11] = 3 (5+5+1)
```

### Brute Force Alternative: O(amount^coins)

```csharp
// Try all combinations recursively
public int CoinChange(int[] coins, int amount)
{
    if (amount == 0) return 0;
    if (amount < 0) return -1;
    
    int min = int.MaxValue;
    foreach (int coin in coins)
    {
        int res = CoinChange(coins, amount - coin);
        if (res >= 0 && res < min)
            min = res + 1;
    }
    
    return min == int.MaxValue ? -1 : min;
}
```

**Why Tabulation is Better:** O(n×coins) vs exponential. Avoids recalculating same amounts repeatedly.

## Strategy 4: 2D DP with Space Optimization

**2D DP** solves problems with two dimensions (two strings, grid paths). The standard approach uses a 2D table where dp[i][j] represents the solution considering first i elements of one dimension and first j of another. Space optimization: observe that each row only depends on the previous row, so reduce 2D array to 1D by reusing space. This drops space from O(m×n) to O(min(m,n)).

### When to use it

- Comparing two sequences (strings, arrays)
- Grid-based problems
- Two-variable subproblems
- Can optimize 2D → 1D space

### Time Complexity: O(m×n) | Space Complexity: O(min(m,n))

### Problem: Longest Common Subsequence

Find length of longest common subsequence between two strings.

**Implementation:**

```csharp
public class Solution
{
    public int LongestCommonSubsequence(string text1, string text2)
    {
        int m = text1.Length, n = text2.Length;
        
        // Optimize: use smaller dimension for DP array
        if (m < n)
        {
            (m, n) = (n, m);
            (text1, text2) = (text2, text1);
        }
        
        int[] dp = new int[n + 1];
        
        for (int i = 0; i < m; i++)
        {
            int prev = 0;
            for (int j = 0; j < n; j++)
            {
                int cur = dp[j + 1];
                if (text1[i] == text2[j])
                    dp[j + 1] = prev + 1;
                else
                    dp[j + 1] = Math.Max(dp[j], dp[j + 1]);
                prev = cur;
            }
        }
        
        return dp[n];
    }
}
```

**Key Insight:** Standard 2D DP: if chars match, add 1 to diagonal. Else, take max of left/top. Space optimization: keep only current and previous row (or use 1D with prev variable).

**Recurrence (2D):**
```
dp[i][j] = dp[i-1][j-1] + 1           if text1[i] == text2[j]
dp[i][j] = max(dp[i-1][j], dp[i][j-1]) otherwise
```

**Example:**
```
text1 = "abcde", text2 = "ace"
    ""  a  c  e
""   0  0  0  0
a    0  1  1  1
b    0  1  1  1
c    0  1  2  2
d    0  1  2  2
e    0  1  2  3
Result: 3 (ace)
```

### Brute Force Alternative: O(2^(m+n))

```csharp
// Try all subsequences
public int LongestCommonSubsequence(string s1, string s2)
{
    return LCS(s1, s2, 0, 0);
}

private int LCS(string s1, string s2, int i, int j)
{
    if (i == s1.Length || j == s2.Length) return 0;
    if (s1[i] == s2[j])
        return 1 + LCS(s1, s2, i+1, j+1);
    return Math.Max(LCS(s1, s2, i+1, j), LCS(s1, s2, i, j+1));
}
```

**Why 2D DP is Better:** O(m×n) vs O(2^(m+n)). Space optimization further reduces to O(min(m,n)).

## Strategy Selection Guide

| Problem Type | DP Strategy | Time | Space |
|-------------|-------------|------|-------|
| Climbing stairs / Fibonacci | Bottom-Up (2 variables) | O(n) | O(1) |
| House robber / Buy-sell stock | State Machine | O(n) | O(1) |
| Coin change / Unbounded knapsack | Tabulation (1D) | O(n×m) | O(n) |
| 0/1 Knapsack | Tabulation (2D→1D) | O(n×W) | O(W) |
| LCS / Edit distance | 2D DP optimized | O(m×n) | O(min(m,n)) |
| Word break | Tabulation + substring | O(n²) | O(n) |
| Longest increasing subsequence | Tabulation or Binary Search | O(n²) or O(n log n) | O(n) |
| Decode ways | Fibonacci pattern | O(n) | O(1) |

## Pattern Recognition Tips

1. **"Maximum/minimum..."** → Likely DP optimization problem
2. **"Count number of ways..."** → Counting DP (sum combinations)
3. **"Longest/shortest subsequence..."** → 2D DP comparing sequences
4. **"Can't use adjacent..."** → State machine with constraints
5. **"Unbounded..."** → Tabulation with inner loop
6. **"0/1 constraint..."** → 2D knapsack pattern
7. **Previous state = f(more previous states)** → DP candidate

## DP Problem-Solving Framework

### Step 1: Identify DP Characteristics
```
✓ Optimal substructure: optimal solution contains optimal sub-solutions
✓ Overlapping subproblems: same subproblems solved repeatedly
✗ Greedy doesn't work: need to consider multiple possibilities
```

### Step 2: Define State
```
State = what you need to know to solve subproblem
Examples:
- dp[i] = answer for first i elements
- dp[i][j] = answer comparing i elements of seq1 with j of seq2
- rob[i], skip[i] = max money if rob/skip house i
```

### Step 3: Write Recurrence Relation
```
How does state[i] relate to previous states?
Examples:
- dp[i] = dp[i-1] + dp[i-2]  (Fibonacci)
- dp[i] = nums[i] + max(dp[i-2], dp[i-3])  (House Robber variant)
- dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]  (Path sum)
```

### Step 4: Determine Base Cases
```
What are the smallest subproblems?
Examples:
- dp[0] = 1, dp[1] = 1  (Fibonacci)
- dp[0] = 0  (Coin change for amount 0)
- dp[i][0] = 0, dp[0][j] = 0  (2D problems with empty sequence)
```

### Step 5: Choose Implementation
```
Top-Down: Natural recursion, easier to write, handles sparse problems
Bottom-Up: Avoids recursion overhead, explicit iteration order
Space Optimization: Often possible if only need recent states
```

## Common DP Optimizations

### Space Optimization
```csharp
// From O(n) to O(1) when only need previous k states
// Before: dp[i] = dp[i-1] + dp[i-2]
int prev = 1, curr = 1;
for (int i = 2; i <= n; i++)
{
    int next = prev + curr;
    prev = curr;
    curr = next;
}

// From O(m×n) to O(n) in 2D DP
// Keep only current and previous row
```

### Memoization vs Tabulation
```csharp
// Top-Down (Memoization)
int[] memo = new int[n];
Array.Fill(memo, -1);

int DP(int i)
{
    if (i <= 1) return 1;
    if (memo[i] != -1) return memo[i];
    return memo[i] = DP(i-1) + DP(i-2);
}

// Bottom-Up (Tabulation)
int[] dp = new int[n+1];
dp[0] = dp[1] = 1;
for (int i = 2; i <= n; i++)
    dp[i] = dp[i-1] + dp[i-2];
```

## Conclusion

Dynamic Programming is powerful but requires practice to master. The key is to:

1. **Recognize DP problems** - Overlapping subproblems + optimal substructure
2. **Define state clearly** - What information do you need?
3. **Write recurrence relation** - How do states relate?
4. **Choose approach** - Top-down vs bottom-up
5. **Optimize space** - Often possible to reduce dimensions

Start with simple Fibonacci-style problems, then progress to state machines, then 2D DP. With practice, you'll quickly identify DP patterns and know which strategy to apply. 💡

## References

_LeetCode. (2024). Dynamic Programming Problems. https://leetcode.com/tag/dynamic-programming/_

_Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press._

_Kleinberg, J., & Tardos, É. (2005). Algorithm Design. Pearson._

_Skiena, S. S. (2008). The Algorithm Design Manual (2nd ed.). Springer._
