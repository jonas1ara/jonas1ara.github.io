---
title: Algorithmic Strategies for Strings 📝
description: Master the essential algorithmic strategies to solve string problems in LeetCode. Learn sliding window, two pointers, hash maps, and stack-based techniques with real C# implementations.
Author: Jonas Lara
date: 2026-01-06 00:00:00 +0000
categories: [Algorithms, Data Structures, LeetCode]
tags: [strings, sliding window, two pointers, hash map, stack, palindrome]
image:
  path: /assets/img/post/algorithmic-strategies-for-strings/strings.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/algorithmic-strategies-for-strings/strings.png
  alt:  Concatenation of two fragments
---

## Introduction

Strings are one of the most common data types in programming and appear frequently in coding interviews. **The key to solving string problems efficiently is recognizing patterns like sliding windows, character frequency counting, and pointer manipulation**. This guide explores the most common strategies with real C# implementations from LeetCode problems.

## Common Algorithmic Strategies

When approaching string problems, you'll typically encounter these strategies:

- **Sliding Window**: Maintain a dynamic window over the string
- **Two Pointers**: Use pointers from both ends or at different speeds
- **Hash Map/Set**: Track character frequencies or seen characters
- **Stack**: Process nested structures or matching pairs
- **String Manipulation**: Sorting, reversing, or transforming strings

## Strategy 1: Sliding Window

**Sliding Window** is a technique for processing sequences by maintaining a window (subarray/substring) that slides through the data. The window can expand or contract based on conditions, allowing O(n) solutions for problems that might otherwise require O(n²). It's particularly effective for finding optimal substrings, subarrays, or sequences that satisfy certain constraints. There are two types: fixed-size windows and variable-size windows.

### When to use it

- Find longest/shortest substring with certain properties
- Need to track contiguous sequence statistics
- Optimize O(n²) nested loops to O(n)
- Problems involving "subarray" or "substring" optimization

### Time Complexity: O(n) | Space Complexity: O(1) to O(k) where k is alphabet size

### Problem: Longest Substring Without Repeating Characters

Find the length of the longest substring without repeating characters.

**Implementation:**

```csharp
public class Solution
{
    public int LengthOfLongestSubstring(string s)
    {
        if (string.IsNullOrEmpty(s))
            return 0;

        var map = new Dictionary<int, int>();
        var maxLen = 0;
        var lastRepeatPos = -1;

        for (int i = 0; i < s.Length; i++)
        {
            // If character repeats, update last repeat position
            if (map.ContainsKey(s[i]) && lastRepeatPos < map[s[i]])
                lastRepeatPos = map[s[i]];
            
            // Update max length
            if (maxLen < i - lastRepeatPos)
                maxLen = i - lastRepeatPos;
            
            map[s[i]] = i;
        }

        return maxLen;
    }
}
```

**Key Insight:** Use a hash map to store the last seen index of each character. When a repeat is found, move the window start to just after the previous occurrence. This avoids nested loops.

**Example:**
```
Input: "abcabcbb"
Process: a(0) → ab(1) → abc(2) → bc(3,repeat a) → bca(4) → cab(5) → abc(6) → bc(7) → c(8)
Output: 3 (substring "abc")
```

### Problem: Longest Repeating Character Replacement

Find the length of the longest substring containing the same letter after replacing at most k characters.

**Implementation:**

```csharp
public class Solution
{
    public int CharacterReplacement(string s, int k)
    {
        int i = 0, j = 0;
        int[] cnt = new int[26];
        int n = s.Length;
        
        while (j < n)
        {
            cnt[s[j] - 'A']++;
            j++;
            
            // If window invalid (more than k replacements needed), shrink from left
            if (j - i - MaxElement(cnt, 26) > k)
            {
                cnt[s[i] - 'A']--;
                i++;
            }
        }
        
        return j - i;
    }

    private int MaxElement(int[] arr, int length)
    {
        int max = arr[0];
        for (int i = 1; i < length; i++)
        {
            if (arr[i] > max)
                max = arr[i];
        }
        return max;
    }
}
```

**Key Insight:** Window size = (most frequent char count) + k. Expand window while valid, shrink from left when invalid. The window size represents valid substring length.

### Brute Force Alternative: O(n²)

```csharp
// Check all substrings
public int LengthOfLongestSubstring(string s)
{
    int maxLen = 0;
    
    for (int i = 0; i < s.Length; i++)
    {
        HashSet<char> seen = new HashSet<char>();
        for (int j = i; j < s.Length; j++)
        {
            if (seen.Contains(s[j]))
                break;
            seen.Add(s[j]);
            maxLen = Math.Max(maxLen, j - i + 1);
        }
    }
    
    return maxLen;
}
```

**Why Sliding Window is Better:** Reduces O(n²) nested loops to O(n) single pass by maintaining window state incrementally.

## Strategy 2: Hash Map for Character Counting

**Hash Map (Dictionary) for Character Counting** uses key-value pairs to track character frequencies, enabling O(1) lookup and update operations. This technique is fundamental for anagram detection, character frequency analysis, and substring problems. In C#, `Dictionary<char, int>` or simple arrays (`int[26]` for lowercase letters) serve this purpose efficiently.

### When to use it

- Count character frequencies
- Detect anagrams or permutations
- Check if two strings have same characters
- Track character occurrences in substrings

### Time Complexity: O(n) to O(n·k log k) | Space Complexity: O(1) to O(n)

### Problem: Valid Anagram

Given two strings s and t, return true if t is an anagram of s.

**Implementation:**

```csharp
public class Solution
{
    public bool IsAnagram(string s, string t)
    {
        int[] cnt = new int[26];

        // Increment for s
        foreach (char c in s)
            cnt[c - 'a']++;

        // Decrement for t
        foreach (char c in t)
            cnt[c - 'a']--;

        // Check if all counts are zero
        foreach (int n in cnt)
        {
            if (n != 0)
                return false;
        }

        return true;
    }
}
```

**Key Insight:** Increment counters for first string, decrement for second. If all counters return to zero, strings are anagrams. Using array instead of dictionary is faster for limited alphabet.

### Problem: Group Anagrams

Group strings that are anagrams of each other.

**Implementation:**

```csharp
public class Solution
{
    public IList<IList<string>> GroupAnagrams(string[] strs)
    {
        Dictionary<string, int> m = new Dictionary<string, int>();
        List<IList<string>> ans = new List<IList<string>>();

        for (int i = 0; i < strs.Length; i++)
        {
            // Use sorted string as key
            char[] keyChars = strs[i].ToCharArray();
            Array.Sort(keyChars);
            string key = new string(keyChars);

            if (!m.ContainsKey(key))
            {
                m[key] = ans.Count;
                ans.Add(new List<string>());
            }

            ans[m[key]].Add(strs[i]);
        }

        return ans;
    }
}
```

**Key Insight:** Sorted string serves as canonical form for anagrams. All anagrams map to same sorted key, enabling O(1) grouping with hash map.

**Example:**
```
Input: ["eat","tea","tan","ate","nat","bat"]
Keys: "aet", "aet", "ant", "aet", "ant", "abt"
Output: [["eat","tea","ate"],["tan","nat"],["bat"]]
```

### Brute Force Alternative: O(n²·k)

```csharp
// Compare each string with every other string
public bool IsAnagram(string s, string t)
{
    if (s.Length != t.Length) return false;
    
    char[] sArr = s.ToCharArray();
    char[] tArr = t.ToCharArray();
    Array.Sort(sArr);
    Array.Sort(tArr);
    
    return new string(sArr) == new string(tArr);
}
```

**Why Hash Map is Better:** Direct O(n) comparison vs O(n log n) sorting. For group anagrams, avoids O(n²) pairwise comparisons.

## Strategy 3: Stack for Matching Pairs

**Stack** is a Last-In-First-Out (LIFO) data structure perfect for problems involving nested structures, matching pairs, or reverse processing. For string problems, stacks excel at validating balanced parentheses, processing nested expressions, and handling operations that require backtracking to previous states. The key is recognizing when you need to "remember" previous elements in reverse order.

### When to use it

- Match opening and closing brackets/parentheses
- Process nested structures
- Validate balanced expressions
- Need LIFO (Last In First Out) processing

### Time Complexity: O(n) | Space Complexity: O(n)

### Problem: Valid Parentheses

Determine if a string containing brackets is valid (every opening bracket has matching closing bracket in correct order).

**Implementation:**

```csharp
public class Solution
{
    public bool IsValid(string s)
    {
        Stack<char> stack = new Stack<char>();
        
        foreach (char c in s)
        {
            if (c == '(' || c == '{' || c == '[')
            {
                stack.Push(c);
            }
            else if (stack.Count == 0 || 
                     (c == ')' && stack.Peek() != '(') ||
                     (c == '}' && stack.Peek() != '{') ||
                     (c == ']' && stack.Peek() != '['))
            {
                return false;
            }
            else
            {
                stack.Pop();
            }
        }
        
        return stack.Count == 0;
    }
}
```

**Key Insight:** Push opening brackets onto stack. For closing brackets, check if stack top matches. Valid string leaves empty stack. This ensures proper nesting.

**Example:**
```
Input: "{[()]}"
Stack: { → {[ → {[( → {[ → { → empty ✓

Input: "{[(]}"
Stack: { → {[ → {[( → Error: ] doesn't match ( ✗
```

### Brute Force Alternative: O(n²)

```csharp
// Repeatedly remove valid pairs until no more can be removed
public bool IsValid(string s)
{
    while (s.Contains("()") || s.Contains("{}") || s.Contains("[]"))
    {
        s = s.Replace("()", "").Replace("{}", "").Replace("[]", "");
    }
    return s.Length == 0;
}
```

**Why Stack is Better:** O(n) single pass vs O(n²) repeated string replacements. Stack naturally models the nested structure.

## Strategy 4: Two Pointers

**Two Pointers** uses two indices that traverse the string from different positions - typically from both ends moving inward, or both from the start at different speeds. This technique eliminates the need for nested loops in many scenarios, reducing complexity from O(n²) to O(n). It's particularly effective for palindrome detection, string reversal, and partition problems.

### When to use it

- Check for palindromes
- Reverse or rearrange strings in-place
- Remove/skip certain characters
- Compare strings from both ends

### Time Complexity: O(n) | Space Complexity: O(1)

### Problem: Valid Palindrome

Given a string, determine if it's a palindrome considering only alphanumeric characters and ignoring cases.

**Implementation:**

```csharp
public class Solution
{
    public bool IsPalindrome(string s)
    {
        int i = 0, j = s.Length - 1;
        
        while (i < j)
        {
            // Skip non-alphanumeric from left
            while (i < j && !char.IsLetterOrDigit(s[i]))
                i++;
            
            // Skip non-alphanumeric from right
            while (i < j && !char.IsLetterOrDigit(s[j]))
                j--;
            
            // Compare characters (case-insensitive)
            if (i < j && char.ToLower(s[i]) != char.ToLower(s[j]))
                return false;

            i++;
            j--;
        }

        return true;
    }
}
```

**Key Insight:** Move pointers from both ends toward center, skipping non-alphanumeric characters. If all comparisons match, it's a palindrome. No extra space needed.

**Example:**
```
Input: "A man, a plan, a canal: Panama"
Clean: "amanaplanacanalpanama"
Compare: a==a, m==m, a==a, ... → true
```

### Brute Force Alternative: O(n)

```csharp
// Create cleaned string, then compare with reverse
public bool IsPalindrome(string s)
{
    string clean = "";
    foreach (char c in s)
    {
        if (char.IsLetterOrDigit(c))
            clean += char.ToLower(c);
    }
    
    char[] arr = clean.ToCharArray();
    Array.Reverse(arr);
    return clean == new string(arr);
}
```

**Why Two Pointers is Better:** O(1) space vs O(n) for creating cleaned string and reversed copy. More memory efficient.

## Strategy Selection Guide

| Problem Type | Strategy | Time | Space |
|-------------|----------|------|-------|
| Longest substring without repeats | Sliding Window | O(n) | O(k) |
| Character replacement in substring | Sliding Window | O(n) | O(26) |
| Valid anagram | Hash Map (Array) | O(n) | O(1) |
| Group anagrams | Hash Map + Sort | O(n·k log k) | O(n·k) |
| Valid parentheses | Stack | O(n) | O(n) |
| Valid palindrome | Two Pointers | O(n) | O(1) |
| Minimum window substring | Sliding Window + Hash | O(n) | O(k) |
| Longest palindromic substring | Expand Around Center | O(n²) | O(1) |

## Pattern Recognition Tips

1. **"Longest/shortest substring..."** → Think Sliding Window
2. **"Anagram" or "permutation"** → Think Hash Map for frequency counting
3. **"Valid parentheses/brackets"** → Think Stack
4. **"Palindrome"** → Think Two Pointers or Expand Around Center
5. **"Without repeating"** → Think Sliding Window + Hash Set
6. **"Match/find pattern"** → Think Hash Map or KMP algorithm
7. **"Encode/decode"** → Think String Manipulation or Delimiter-based

## Common String Operations in C#

### Character Manipulation
```csharp
// Check character type
char.IsLetter(c)
char.IsDigit(c)
char.IsLetterOrDigit(c)
char.IsUpper(c)
char.IsLower(c)

// Convert case
char.ToLower(c)
char.ToUpper(c)

// ASCII value
int value = (int)c;
char ch = (char)value;
```

### String to Array Conversions
```csharp
// String to char array
char[] arr = str.ToCharArray();

// Array to string
string str = new string(arr);

// Sort characters
Array.Sort(arr);
```

### StringBuilder for Efficiency
```csharp
// Use StringBuilder for multiple concatenations (O(1) amortized)
StringBuilder sb = new StringBuilder();
sb.Append("text");
string result = sb.ToString();
```

## Conclusion

Mastering string algorithms is essential for coding interviews and real-world applications. The key is to:

1. **Identify the pattern** (sliding window, palindrome, matching, etc.)
2. **Choose the right data structure** (hash map, stack, two pointers)
3. **Optimize space** when possible (in-place manipulation, arrays vs maps)
4. **Handle edge cases** (empty strings, single characters, special characters)

Practice these strategies on LeetCode to build pattern recognition. With experience, you'll quickly identify the optimal approach for any string problem. 📝

## References

_LeetCode. (2024). String Problems. https://leetcode.com/tag/string/_

_Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press._

_Knuth, D. E., Morris, J. H., & Pratt, V. R. (1977). Fast Pattern Matching in Strings. SIAM Journal on Computing._
