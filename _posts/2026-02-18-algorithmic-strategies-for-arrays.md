---
title: Algorithmic Strategies for Arrays 🧩
description: Master the most common algorithmic strategies to solve array problems in LeetCode. Learn how to identify patterns and apply the right approach with real C# implementations.
Author: Jonas Lara
date: 2026-02-18 00:00:00 +0000
categories: [Algorithms, Data Structures, LeetCode]
tags: [arrays, hash table, two pointers, dynamic programming, binary search, kadane, prefix sum]
image:
  path: /assets/img/post/algorithmic-strategies-for-arrays/arrays-kp.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/algorithmic-strategies-for-arrays/arrays-kp.png
  alt: Knapsack Problem
---

## Introduction

Arrays are one of the most fundamental data structures in computer science and are frequently featured in coding interviews and LeetCode problems. **The key to solving array problems efficiently is recognizing patterns and applying the right algorithmic strategy**. This guide explores the most common strategies with real C# implementations from LeetCode problems.

## Common Algorithmic Strategies

When approaching array problems, you'll typically encounter one of these strategies:

- **Hash Table**: Use auxiliary space to achieve O(n) time complexity
- **Two Pointers**: Efficiently traverse arrays with multiple pointers
- **Dynamic Programming**: Build solutions from subproblems
- **Binary Search**: Divide and conquer on sorted arrays
- **Kadane's Algorithm**: Find maximum subarray sums
- **Prefix/Postfix**: Pre-compute cumulative values for efficient queries
- **Brute Force**: Try all possible combinations (often O(n²) or worse)

## Strategy 1: Hash Table

A **Hash Table** (also known as Hash Map or Dictionary) is a data structure that stores key-value pairs and provides O(1) average-case lookup, insertion, and deletion operations. It works by using a hash function to compute an index into an array of buckets, from which the desired value can be found. In C#, this is implemented as `Dictionary<TKey, TValue>` or `HashSet<T>` for unique values only.

### When to use it

- Need to find pairs or complements that sum to a target
- Check for duplicates or unique elements
- Need O(1) lookup time
- Can trade space for time

### Time Complexity: O(n) | Space Complexity: O(n)

### Problem: Two Sum

Given an array of integers `nums` and an integer `target`, return indices of two numbers that add up to `target`.

**Implementation:**

```csharp
public class Solution
{
    public int[] TwoSum(int[] nums, int target)
    {
        var dic = new Dictionary<int, int>();

        for (int i = 0; i < nums.Length; i++)
        {
            int complement = target - nums[i];

            if (dic.ContainsKey(complement)) 
                return new int[] { dic[complement], i };

            dic[nums[i]] = i;
        }

        return new int[] { };
    }
}
```

**Key Insight:** Instead of checking every pair (O(n²)), we store each number and its index in a hash table, then check if the complement exists.

### Problem: Contains Duplicate

Given an integer array `nums`, return `true` if any value appears at least twice.

**Implementation:**

```csharp
public class Solution
{
    public bool ContainsDuplicate(int[] nums)
    {
        HashSet<int> set = new HashSet<int>(nums);
        return set.Count != nums.Length;
    }
}
```

**Key Insight:** HashSet automatically handles uniqueness. If the set size differs from array length, duplicates exist.

## Strategy 2: Two Pointers

The **Two Pointers** technique uses two pointers (or indices) that traverse the array from different positions - typically one from the start and one from the end, or both moving in the same direction at different speeds. This strategy eliminates the need for nested loops and extra space in many scenarios. It's particularly effective when the array is sorted or when you need to find pairs/triplets that satisfy certain conditions.

### When to use it

- Array is sorted or can be sorted
- Need to find pairs/triplets with specific properties
- Optimize space by avoiding extra data structures
- Need to move from both ends towards center

### Time Complexity: O(n) or O(n²) | Space Complexity: O(1)

### Problem: Container With Most Water

Find two lines that together with the x-axis form a container that holds the most water.

**Implementation:**

```csharp
public class Solution
{
    public int MaxArea(int[] height)
    {
        int ans = 0, left = 0, right = height.Length - 1;

        while (left < right)
        {
            ans = Math.Max(ans, (right - left) * Math.Min(height[left], height[right]));
            
            if (height[left] < height[right])
                left++;
            else
                right--;
        }

        return ans;
    }
}
```

**Key Insight:** Start with widest container. Move the pointer with smaller height inward, as moving the larger height can only decrease the area.

### Problem: 3Sum

Find all unique triplets that sum to zero.

**Implementation:**

```csharp
public class Solution
{
    public IList<IList<int>> ThreeSum(int[] nums)
    {
        List<IList<int>> result = new List<IList<int>>();
        int n = nums.Length;

        if (n < 3) return result;

        Array.Sort(nums);

        for (int i = 0; i < n - 2; i++)
        {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i - 1] == nums[i]) continue;

            int j = i + 1;
            int k = n - 1;

            while (j < k)
            {
                int sum = nums[i] + nums[j] + nums[k];

                if (sum < 0)
                    j++;
                else if (sum > 0)
                    k--;
                else
                {
                    result.Add(new List<int> { nums[i], nums[j], nums[k] });

                    while (j < k && nums[j] == nums[j + 1]) j++;
                    while (j < k && nums[k - 1] == nums[k]) k--;
                    
                    j++;
                    k--;
                }
            }
        }
        
        return result;
    }
}
```

**Key Insight:** Fix one element, then use two pointers to find pairs that sum to the negative of the fixed element. Sorting helps avoid duplicates.

### Problem: Maximum Product Subarray

Find the contiguous subarray with the largest product.

**Implementation:**

```csharp
public class Solution
{
    public int MaxProduct(int[] nums)
    {
        int ans = nums[0];
        int N = nums.Length;
        int j = 0;

        while (j < N)
        {
            int i = j;
            int prod = 1;

            while (j < N && nums[j] != 0)
            {
                prod *= nums[j++];
                ans = Math.Max(ans, prod);
            }

            if (j < N)
                ans = Math.Max(ans, 0);

            while (i < N && prod < 0)
            {
                prod /= nums[i++];
                if (i != j)
                    ans = Math.Max(ans, prod);
            }

            while (j < N && nums[j] == 0)
                j++;
        }

        return ans;
    }
}
```

**Key Insight:** Handle zeros as array separators. For negative products, try removing elements from the left to potentially get a positive product.

## Strategy 3: Kadane's Algorithm

**Kadane's Algorithm** is a dynamic programming approach specifically designed to find the maximum sum of a contiguous subarray in O(n) time. The algorithm maintains a running sum and makes a greedy choice at each position: either extend the current subarray or start a new one. This elegant technique was discovered by Jay Kadane in 1984 and has become a classic algorithm for solving maximum subarray problems.

### When to use it

- Finding maximum/minimum subarray sum
- Problems involving contiguous subarrays
- Need O(n) solution instead of O(n²) brute force

### Time Complexity: O(n) | Space Complexity: O(1)

### Problem: Maximum Subarray

Find the contiguous subarray with the largest sum.

**Implementation:**

```csharp
public class Solution
{
    public int MaxSubArray(int[] nums)
    {
        int ans = nums[0];

        for (int i = 1; i < nums.Length; i++)
        {
            nums[i] = nums[i] + Math.Max(nums[i - 1], 0);
            ans = Math.Max(ans, nums[i]);
        }

        return ans;
    }
}
```

**Key Insight:** At each position, decide whether to extend the previous subarray or start fresh. If the previous sum is negative, starting fresh is better.

**Kadane's Algorithm Formula:**
```
currentSum = max(currentElement, currentElement + previousSum)
maxSum = max(maxSum, currentSum)
```

## Strategy 4: Dynamic Programming

**Dynamic Programming (DP)** is a powerful algorithmic paradigm that solves complex problems by breaking them down into simpler overlapping subproblems and storing their solutions to avoid redundant computations. It works on the principle of optimal substructure: the optimal solution to a problem contains optimal solutions to its subproblems. DP can be implemented using either top-down (memoization) or bottom-up (tabulation) approaches, and is essential for optimization problems where you need to track multiple states.

### When to use it

- Problem has optimal substructure
- Overlapping subproblems exist
- Need to track states across iterations
- Making sequential decisions

### Time Complexity: O(n) to O(n²) | Space Complexity: O(1) to O(n)

### Problem: Best Time to Buy and Sell Stock

Find the maximum profit from buying and selling a stock once.

**Implementation:**

```csharp
public class Solution
{
    public int MaxProfit(int[] prices)
    {
        int buy = int.MinValue, sell = 0;

        foreach (int price in prices)
        {
            buy = Math.Max(buy, -price);
            sell = Math.Max(sell, buy + price);
        }

        return sell;
    }
}
```

**Key Insight:** Track two states: maximum profit after buying and maximum profit after selling. Update both states at each price point.

## Strategy 5: Prefix/Postfix

**Prefix/Postfix** (also known as Prefix Sum or Cumulative Sum) is a preprocessing technique where you compute cumulative values from left to right (prefix) and/or right to left (postfix) through an array. This allows you to answer range queries or compute values that depend on all other elements in O(1) or O(n) time instead of O(n²). The prefix array at index i contains the cumulative result of all elements from 0 to i, while postfix contains results from i to n-1.

### When to use it

- Need to compute products/sums excluding current element
- Problems involving cumulative operations
- Can't use division or need to avoid it
- Need O(n) solution for range queries

### Time Complexity: O(n) | Space Complexity: O(1) to O(n)

### Problem: Product of Array Except Self

Return an array where each element is the product of all other elements.

**Implementation:**

```csharp
public class Solution
{
    public int[] ProductExceptSelf(int[] nums)
    {
        var result = new int[nums.Length];
        result[0] = 1;

        // Build prefix products
        for (int i = 1; i < nums.Length; i++)
        {
            result[i] = result[i - 1] * nums[i - 1];
        }

        // Multiply by postfix products
        int rightSide = 1;
        for (int i = nums.Length - 1; i >= 0; i--)
        {
            result[i] = result[i] * rightSide;
            rightSide *= nums[i];
        }

        return result;
    }
}
```

**Key Insight:** First pass stores prefix products. Second pass multiplies by postfix products on-the-fly. This avoids division and handles zeros naturally.

## Strategy 6: Binary Search

**Binary Search** is a divide-and-conquer algorithm that efficiently searches for a target value in a sorted array by repeatedly dividing the search interval in half. It compares the target with the middle element and eliminates half of the remaining elements based on the comparison. This logarithmic time complexity makes it dramatically faster than linear search for large datasets. Binary search requires the data to be sorted (or have some monotonic property) to work correctly.

### When to use it

- Array is sorted (or rotated sorted)
- Need to find element in O(log n) time
- Search space can be halved each iteration
- Finding minimum/maximum in sorted structure

### Time Complexity: O(log n) | Space Complexity: O(1)

### Problem: Find Minimum in Rotated Sorted Array

Find the minimum element in a rotated sorted array.

**Implementation:**

```csharp
public class Solution
{
    public int FindMin(int[] nums)
    {
        int left = 0, right = nums.Length - 1;

        while (left < right)
        {
            int m = (left + right) / 2;

            if (nums[m] > nums[right])
                left = m + 1;
            else
                right = m;
        }

        return nums[left];
    }
}
```

**Key Insight:** Compare middle element with rightmost. If middle is greater, minimum is in right half; otherwise, it's in left half (including middle).

### Problem: Search in Rotated Sorted Array

Search for a target value in a rotated sorted array.

**Implementation:**

```csharp
public class Solution
{
    public int Search(int[] nums, int target)
    {
        if (nums.Length == 0) return -1;

        int n = nums.Length, left = 0, right = n - 1, pivot;

        // Find rotation pivot
        while (left < right)
        {
            int m = left + (right - left) / 2;

            if (nums[m] < nums[right])
                right = m;
            else
                left = m + 1;
        }
        
        pivot = left;
        left = 0;
        right = n - 1;

        // Binary search with pivot offset
        while (left <= right)
        {
            int m = left + (right - left) / 2;
            int mm = (m + pivot) % n;

            if (nums[mm] == target)
                return mm;
            if (target > nums[mm])
                left = m + 1;
            else
                right = m - 1;
        }

        return -1;
    }
}
```

**Key Insight:** First find the rotation point, then perform binary search with modulo arithmetic to handle the rotation.

## Strategy 7: Brute Force

**Brute Force** is the most straightforward problem-solving approach that tries all possible solutions without any optimization. It typically involves nested loops that exhaustively check every combination or possibility until finding the answer. While simple to implement and understand, brute force solutions often have poor time complexity (O(n²), O(n³), or worse) and are not suitable for large inputs. However, they serve as an excellent starting point for understanding a problem and establishing correctness before optimization.

### When to use it

- Problem size is small (n ≤ 100)
- Need to establish correctness before optimizing
- No better algorithm is known
- Baseline for comparison

### Time Complexity: O(n²) to O(n³) | Space Complexity: O(1)

**Note:** Brute force solutions are often the starting point but should be optimized for larger inputs. Most of the problems above have brute force alternatives (nested loops checking all pairs/triplets) that would time out on LeetCode.

## Strategy Selection Guide

| Problem Type | Strategy | Time | Space |
|-------------|----------|------|-------|
| Find pair with sum | Hash Table | O(n) | O(n) |
| Find triplet with sum | Two Pointers | O(n²) | O(1) |
| Maximum subarray sum | Kadane's | O(n) | O(1) |
| Buy/sell stock | Dynamic Programming | O(n) | O(1) |
| Product except self | Prefix/Postfix | O(n) | O(1) |
| Search in sorted array | Binary Search | O(log n) | O(1) |
| Check duplicates | Hash Table | O(n) | O(n) |
| Container with water | Two Pointers | O(n) | O(1) |

## Pattern Recognition Tips

1. **Sorted array?** → Think Binary Search or Two Pointers
2. **Find pairs/complements?** → Think Hash Table
3. **Contiguous subarray?** → Think Kadane's or DP
4. **Product/sum excluding element?** → Think Prefix/Postfix
5. **Multiple states to track?** → Think Dynamic Programming
6. **Move from both ends?** → Think Two Pointers

## Conclusion

Mastering these algorithmic strategies is essential for solving array problems efficiently. The key is to:

1. **Identify the pattern** in the problem statement
2. **Choose the right strategy** based on constraints
3. **Implement cleanly** with proper edge case handling
4. **Analyze complexity** to ensure scalability

Practice these strategies on LeetCode to build intuition for which approach to use. With time, pattern recognition becomes second nature, and you'll quickly identify the optimal strategy for any array problem you encounter. 🚀

## References

_LeetCode. (2024). Array Problems. https://leetcode.com/tag/array/_

_Cormen, T. H., Leiserson, C. E., Rivert, R. L. & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press._

_Kadane, J. (1984). Maximum Subarray Problem Algorithm._
