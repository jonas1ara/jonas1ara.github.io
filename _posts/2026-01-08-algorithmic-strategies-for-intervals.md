---
title: Algorithmic Strategies for Intervals ⏱️
description: Master the essential algorithmic strategies to solve interval problems in LeetCode. Learn sorting, greedy algorithms, and merging techniques with real C# implementations.
Author: Jonas Lara
date: 2026-01-08 00:00:00 +0000
categories: [Algorithms, Data Structures, LeetCode]
tags: [intervals, sorting, greedy, merge, scheduling]
image:
  path: /assets/img/post/algorithmic-strategies-intervals/intervals.jpg
  lqip: https://images.unsplash.com/photo-1501139083538-0139583c060f?q=80&w=1770&auto=format&fit=crop
  alt: Interval Algorithms
---

## Introduction

Interval problems are common in scheduling, time management, and resource allocation scenarios. **The key to solving interval problems is recognizing that sorting by start or end time often enables greedy or linear solutions**. This guide explores the most common strategies with real C# implementations from LeetCode problems.

## Common Algorithmic Strategies

When approaching interval problems, you'll typically encounter these strategies:

- **Sorting + Merging**: Sort intervals then merge overlapping ones
- **Greedy Selection**: Make locally optimal choices for global optimum
- **Linear Scan**: Process sorted intervals in one pass
- **Two Pointers**: Track multiple interval boundaries simultaneously

## Strategy 1: Sort and Merge

**Sort and Merge** is the fundamental technique for interval problems. By sorting intervals by start time, overlapping intervals become adjacent, enabling efficient merging in a single pass. This transforms a potentially O(n²) comparison problem into O(n log n) sorting plus O(n) merging. The key insight is that after sorting, you only need to compare each interval with the most recently merged interval.

### When to use it

- Need to merge overlapping intervals
- Combine or consolidate time ranges
- Detect interval conflicts
- Simplify interval representation

### Time Complexity: O(n log n) | Space Complexity: O(n)

### Problem: Merge Intervals

Given an array of intervals, merge all overlapping intervals.

**Implementation:**

```csharp
public class Solution
{
    public int[][] Merge(int[][] intervals)
    {
        if (intervals.Length == 0)
            return new int[0][];

        // Sort by start time
        Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));

        List<int[]> merged = new List<int[]>();
        merged.Add(intervals[0]);

        for (int i = 1; i < intervals.Length; i++)
        {
            // If current interval doesn't overlap with last merged
            if (merged.Last()[1] < intervals[i][0])
            {
                merged.Add(intervals[i]);
            }
            else
            {
                // Merge by extending the end time
                merged.Last()[1] = Math.Max(merged.Last()[1], intervals[i][1]);
            }
        }

        return merged.ToArray();
    }
}
```

**Key Insight:** After sorting by start time, overlapping intervals are adjacent. Compare each interval with the last merged one: if they overlap (current.start ≤ last.end), extend the end; otherwise, add as new interval.

**Example:**
```
Input: [[1,3],[2,6],[8,10],[15,18]]
After sort: [[1,3],[2,6],[8,10],[15,18]]
Process:
  [1,3] - add
  [2,6] - overlaps with [1,3] → merge to [1,6]
  [8,10] - no overlap → add
  [15,18] - no overlap → add
Output: [[1,6],[8,10],[15,18]]
```

### Brute Force Alternative: O(n²)

```csharp
// Compare every interval with every other interval
public int[][] Merge(int[][] intervals)
{
    bool[] merged = new bool[intervals.Length];
    List<int[]> result = new List<int[]>();
    
    for (int i = 0; i < intervals.Length; i++)
    {
        if (merged[i]) continue;
        
        int start = intervals[i][0];
        int end = intervals[i][1];
        
        for (int j = i + 1; j < intervals.Length; j++)
        {
            if (!(intervals[j][1] < start || intervals[j][0] > end))
            {
                start = Math.Min(start, intervals[j][0]);
                end = Math.Max(end, intervals[j][1]);
                merged[j] = true;
            }
        }
        
        result.Add(new int[] {start, end});
    }
    
    return result.ToArray();
}
```

**Why Sort and Merge is Better:** Reduces O(n²) pairwise comparisons to O(n log n) sorting + O(n) linear scan. Much more efficient for large inputs.

## Strategy 2: Linear Insertion

**Linear Insertion** handles inserting a new interval into a sorted, non-overlapping list of intervals. Since intervals are already sorted, we can process them in one pass, determining whether to merge with the new interval or add independently. This avoids re-sorting the entire list and achieves O(n) time complexity by leveraging the existing order.

### When to use it

- Insert new interval into sorted list
- Already have non-overlapping intervals
- Need to maintain sorted order
- Avoid re-sorting entire list

### Time Complexity: O(n) | Space Complexity: O(n)

### Problem: Insert Interval

Insert a new interval into a sorted list of non-overlapping intervals and merge if necessary.

**Implementation:**

```csharp
public class Solution
{
    public int[][] Insert(int[][] intervals, int[] newInterval)
    {
        List<int[]> ans = new List<int[]>();
        int start = newInterval[0];
        int end = newInterval[1];

        foreach (int[] interval in intervals)
        {
            if (start > end)
            {
                // New interval already inserted, add remaining intervals
                ans.Add(interval);
            }
            else if (interval[1] < start)
            {
                // Current interval before new interval
                ans.Add(interval);
            }
            else if (interval[0] > end)
            {
                // Current interval after new interval - insert new and continue
                ans.Add(new int[] { start, end });
                start = end + 1; // Mark as inserted
                ans.Add(interval);
            }
            else
            {
                // Overlap - merge by extending boundaries
                start = Math.Min(start, interval[0]);
                end = Math.Max(end, interval[1]);
            }
        }

        // Insert if not yet added
        if (start <= end)
        {
            ans.Add(new int[] { start, end });
        }

        return ans.ToArray();
    }
}
```

**Key Insight:** Process intervals in order. Three cases: (1) before new interval - add as-is, (2) overlaps - extend merge boundaries, (3) after - insert merged interval then add rest. Single pass achieves O(n).

**Example:**
```
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Process:
  [1,2] - before [4,8] → add
  [3,5] - overlaps → merge boundaries to [3,8]
  [6,7] - overlaps → extend to [3,8]
  [8,10] - overlaps → extend to [3,10]
  [12,16] - after → insert [3,10], add [12,16]
Output: [[1,2],[3,10],[12,16]]
```

### Brute Force Alternative: O(n log n)

```csharp
// Add new interval and re-sort/merge entire list
public int[][] Insert(int[][] intervals, int[] newInterval)
{
    List<int[]> list = new List<int[]>(intervals);
    list.Add(newInterval);
    return Merge(list.ToArray()); // Use Merge from previous problem
}
```

**Why Linear Insertion is Better:** O(n) vs O(n log n) by avoiding full re-sort. Leverages existing order for efficiency.

## Strategy 3: Greedy Selection

**Greedy Selection** solves interval optimization problems by making locally optimal choices that lead to a globally optimal solution. For intervals, this often means sorting by a specific criterion (start/end time) then iteratively selecting intervals that satisfy constraints. The greedy choice property guarantees that selecting the "best" next interval at each step produces the overall best result.

### When to use it

- Maximize non-overlapping intervals
- Minimize intervals to remove
- Room/resource scheduling
- Activity selection problems

### Time Complexity: O(n log n) | Space Complexity: O(1)

### Problem: Non-overlapping Intervals

Find the minimum number of intervals to remove to make the rest non-overlapping.

**Implementation:**

```csharp
public class Solution
{
    public int EraseOverlapIntervals(int[][] intervals)
    {
        // Sort by end time (greedy choice)
        Array.Sort(intervals, (a, b) => a[1].CompareTo(b[1]));

        int ans = 0;
        int end = int.MinValue;

        foreach (var interval in intervals)
        {
            if (interval[0] >= end)
            {
                // No overlap - keep this interval
                end = interval[1];
            }
            else
            {
                // Overlap - remove this interval
                ans++;
            }
        }

        return ans;
    }
}
```

**Key Insight:** Sort by end time. Greedy strategy: always keep the interval that ends earliest. This leaves maximum room for future intervals. Count overlaps as removals.

**Greedy Proof:** Keeping intervals with earliest end times maximizes available space for subsequent intervals, leading to minimum removals.

**Example:**
```
Input: [[1,2],[2,3],[3,4],[1,3]]
Sort by end: [[1,2],[2,3],[1,3],[3,4]]
Process:
  [1,2] - keep (end=2)
  [2,3] - keep (start≥2, end=3)
  [1,3] - overlap (start<3) → remove
  [3,4] - keep (start≥3)
Output: 1 (removed 1 interval)
```

### Problem: Meeting Rooms

Determine if a person can attend all meetings (no overlapping meetings).

**Implementation:**

```csharp
public class Solution
{
    public bool CanAttendMeetings(List<Interval> intervals)
    {
        // Sort by start time
        intervals.Sort((a, b) => a.start - b.start);

        for (int i = 1; i < intervals.Count; i++)
        {
            // Check if current meeting starts before previous ends
            if (intervals[i].start < intervals[i - 1].end)
            {
                return false; // Overlap detected
            }
        }

        return true;
    }
}
```

**Key Insight:** After sorting by start time, check consecutive intervals. If any meeting starts before the previous one ends, there's a conflict.

### Brute Force Alternative: O(n²)

```csharp
// Check every interval against every other interval
public int EraseOverlapIntervals(int[][] intervals)
{
    int n = intervals.Length;
    int[] dp = new int[n];
    Array.Fill(dp, 1);
    
    // For each interval, count max non-overlapping before it
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (intervals[j][1] <= intervals[i][0])
            {
                dp[i] = Math.Max(dp[i], dp[j] + 1);
            }
        }
    }
    
    return n - dp.Max(); // Total - max non-overlapping
}
```

**Why Greedy is Better:** O(n log n) sorting with O(n) scan vs O(n²) dynamic programming. Greedy's locally optimal choice guarantees global optimum for this problem class.

## Strategy Selection Guide

| Problem Type | Strategy | Time | Space |
|-------------|----------|------|-------|
| Merge overlapping intervals | Sort + Merge | O(n log n) | O(n) |
| Insert into sorted intervals | Linear Scan | O(n) | O(n) |
| Minimum intervals to remove | Greedy (sort by end) | O(n log n) | O(1) |
| Can attend all meetings | Sort + Check consecutive | O(n log n) | O(1) |
| Maximum non-overlapping | Greedy Selection | O(n log n) | O(1) |
| Minimum meeting rooms needed | Sort start/end separately | O(n log n) | O(n) |

## Pattern Recognition Tips

1. **"Merge intervals"** → Sort by start, merge overlapping
2. **"Minimum/maximum non-overlapping"** → Greedy (sort by end time)
3. **"Can attend all meetings"** → Sort and check consecutive
4. **"Insert interval"** → Linear scan (already sorted)
5. **"Number of rooms needed"** → Track simultaneous intervals
6. **Any interval problem** → Start with sorting (usually by start or end)

## Common Interval Patterns

### Interval Definition
```csharp
// Common representation
int[][] intervals; // intervals[i] = [start, end]

// Or custom class
public class Interval
{
    public int start, end;
    public Interval(int start, int end)
    {
        this.start = start;
        this.end = end;
    }
}
```

### Sorting Patterns
```csharp
// Sort by start time
Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));

// Sort by end time (for greedy)
Array.Sort(intervals, (a, b) => a[1].CompareTo(b[1]));

// Sort by custom logic
Array.Sort(intervals, (a, b) => {
    if (a[0] != b[0]) return a[0].CompareTo(b[0]);
    return a[1].CompareTo(b[1]);
});
```

### Overlap Detection
```csharp
// Two intervals overlap if:
bool Overlaps(int[] a, int[] b)
{
    return !(a[1] < b[0] || b[1] < a[0]);
    // Or equivalently: a[0] <= b[1] && b[0] <= a[1]
}
```

## Conclusion

Mastering interval algorithms is essential for scheduling and time management problems. The key is to:

1. **Sort first** - Almost always the first step (by start or end time)
2. **Identify the greedy property** - Does earliest end time work? Earliest start?
3. **Handle edge cases** - Empty arrays, single interval, all overlapping
4. **Choose the right sort criterion** - Start time for merging, end time for greedy selection

Practice these strategies on LeetCode to recognize patterns quickly. Interval problems often have elegant greedy solutions once you identify the right sorting strategy. ⏱️

## References

_LeetCode. (2024). Interval Problems. https://leetcode.com/tag/intervals/_

_Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press._

_Kleinberg, J., & Tardos, É. (2005). Algorithm Design. Pearson._
