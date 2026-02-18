---
title: Algorithmic Strategies for Binary 🔢
description: Master bit manipulation techniques to solve binary problems in LeetCode. Learn XOR tricks, Brian Kernighan's algorithm, and dynamic programming for bits with real C# implementations.
Author: Jonas Lara
date: 2026-01-12 00:00:00 +0000
categories: [Algorithms, Data Structures, LeetCode]
tags: [binary, bit manipulation, xor, brian kernighan, bits, bitwise]
image:
  path: /assets/img/post/algorithmic-strategies-binary/binary.jpg
  lqip: https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=1770&auto=format&fit=crop
  alt: Binary Algorithms
---

## Introduction

Binary and bit manipulation problems require understanding how numbers are represented at the bit level. **The key to solving these problems is recognizing that bitwise operations can often replace arithmetic operations with much faster O(1) solutions**. This guide explores the most common bit manipulation strategies with real C# implementations from LeetCode problems.

## Common Algorithmic Strategies

When approaching binary problems, you'll typically encounter these strategies:

- **Bit Manipulation**: Use bitwise operators (&, |, ^, ~, <<, >>)
- **XOR Properties**: Leverage XOR's unique mathematical properties
- **Brian Kernighan's Algorithm**: Efficiently count set bits
- **Dynamic Programming with Bits**: Build solutions using bit patterns
- **Bit Masking**: Use bits to represent states or sets

## Strategy 1: Brian Kernighan's Algorithm

**Brian Kernighan's Algorithm** is an elegant technique for counting the number of set bits (1s) in a binary number. The key insight is that `n & (n-1)` always flips the rightmost set bit to 0. By repeatedly applying this operation until n becomes 0, we count exactly how many set bits existed. This runs in O(k) time where k is the number of set bits, making it optimal for sparse binary numbers.

### When to use it

- Count number of 1 bits in binary representation
- Need efficient bit counting (Hamming weight)
- Check if number is power of 2
- Find rightmost set bit

### Time Complexity: O(k) where k = number of set bits | Space Complexity: O(1)

### Problem: Number of 1 Bits

Count the number of '1' bits in an unsigned integer (Hamming weight).

**Implementation:**

```csharp
public class Solution
{
    public int HammingWeight(uint n)
    {
        int ans = 0;
        long ln = n;

        while (ln != 0)
        {
            ln -= (ln & -ln); // Remove rightmost set bit
            ans++;
        }

        return ans;
    }
}
```

**Key Insight:** The operation `n & -n` isolates the rightmost set bit. Subtracting it removes that bit. Count iterations until all bits are cleared.

**How it works:**
```
n = 1011 (11 in decimal)
Iteration 1: 1011 & -1011 = 0001, subtract → 1010 (count=1)
Iteration 2: 1010 & -1010 = 0010, subtract → 1000 (count=2)
Iteration 3: 1000 & -1000 = 1000, subtract → 0000 (count=3)
Result: 3 set bits
```

**Alternative approach (same algorithm):**
```csharp
while (n != 0)
{
    n &= (n - 1); // n-1 flips rightmost set bit and all bits to right
    ans++;
}
```

### Brute Force Alternative: O(32) or O(64)

```csharp
// Check every bit position
public int HammingWeight(uint n)
{
    int count = 0;
    for (int i = 0; i < 32; i++)
    {
        if ((n & (1 << i)) != 0)
            count++;
    }
    return count;
}
```

**Why Brian Kernighan is Better:** O(k) where k = set bits vs O(32) checking all positions. For sparse numbers (few 1s), dramatically faster.

## Strategy 2: XOR Properties

**XOR (Exclusive OR)** has unique mathematical properties that make it invaluable for certain problems. Key properties: (1) a ^ a = 0 (self-cancellation), (2) a ^ 0 = a (identity), (3) XOR is commutative and associative. These properties enable finding single unique elements, detecting differences, and swapping without temporary variables. XOR is particularly powerful because it's both reversible and self-inverse.

### When to use it

- Find single unique element when others appear twice
- Swap two variables without temp variable
- Find missing number in sequence
- Detect differences between arrays

### Time Complexity: O(n) | Space Complexity: O(1)

### Problem: Missing Number

Given an array containing n distinct numbers from 0 to n, find the missing number.

**Implementation:**

```csharp
public class Solution
{
    public int MissingNumber(int[] nums)
    {
        int n = nums.Length;
        int xorVal = 0;
        
        for (int i = 0; i < n; i++)
            xorVal ^= nums[i] ^ (i + 1);

        return xorVal;
    }
}
```

**Key Insight:** XOR all array elements with all numbers from 1 to n. Duplicate numbers cancel out (a ^ a = 0), leaving only the missing number.

**Example:**
```
nums = [3, 0, 1] (missing 2)
XOR array: 3 ^ 0 ^ 1 = 2
XOR range: 1 ^ 2 ^ 3 = 0 (complete sequence)
Combined: (3^0^1) ^ (1^2^3) = 3^1^2 = 2 (everything cancels except 2)
```

**XOR Properties in Action:**
```
Properties used:
1. a ^ a = 0 (pairs cancel)
2. a ^ 0 = a (identity)
3. Commutative: a ^ b = b ^ a
4. Associative: (a ^ b) ^ c = a ^ (b ^ c)
```

### Brute Force Alternative: O(n²) or O(n) with extra space

```csharp
// Use HashSet
public int MissingNumber(int[] nums)
{
    HashSet<int> set = new HashSet<int>(nums);
    for (int i = 0; i <= nums.Length; i++)
    {
        if (!set.Contains(i))
            return i;
    }
    return -1;
}

// Or calculate sum difference
public int MissingNumber(int[] nums)
{
    int n = nums.Length;
    int expectedSum = n * (n + 1) / 2;
    int actualSum = nums.Sum();
    return expectedSum - actualSum;
}
```

**Why XOR is Better:** O(1) space vs O(n) for HashSet. Also avoids potential overflow with sum approach for large n.

## Strategy 3: Dynamic Programming with Bits

**Dynamic Programming with Bits** builds solutions by recognizing patterns in binary representations. For counting bits, observe that the bit count of n relates to n/2 (right shift). Specifically, `countBits(n) = countBits(n >> 1) + (n & 1)` - the count equals the count of n/2 plus the last bit. This recurrence enables O(n) computation of all bit counts from 0 to n using memoization.

### When to use it

- Count bits for range of numbers
- Build solutions using previous bit patterns
- Optimize repetitive bit operations
- Need O(n) instead of O(n log n)

### Time Complexity: O(n) | Space Complexity: O(n)

### Problem: Counting Bits

Count the number of 1 bits for every number from 0 to n.

**Implementation:**

```csharp
public class Solution
{
    public int[] CountBits(int n)
    {
        int[] ans = new int[n + 1];

        for (int i = 1; i <= n; i *= 2)
        {
            ans[i] = 1; // Powers of 2 have exactly one bit
            
            // Build from previous power of 2
            for (int j = 1; j < i && i + j <= n; j++)
                ans[i + j] = ans[i] + ans[j];
        }

        return ans;
    }
}
```

**Key Insight:** Powers of 2 have exactly 1 set bit. For other numbers, use pattern: count(i+j) = count(i) + count(j) where i is largest power of 2 ≤ (i+j).

**Pattern Recognition:**
```
0: 0000 → 0 bits
1: 0001 → 1 bit (2^0)
2: 0010 → 1 bit (2^1)
3: 0011 → 2 bits (2 + 1)
4: 0100 → 1 bit (2^2)
5: 0101 → 2 bits (4 + 1)
6: 0110 → 2 bits (4 + 2)
7: 0111 → 3 bits (4 + 3)
```

**Alternative DP formula:**
```csharp
public int[] CountBits(int n)
{
    int[] ans = new int[n + 1];
    for (int i = 1; i <= n; i++)
    {
        ans[i] = ans[i >> 1] + (i & 1);
        // count(i) = count(i/2) + lastBit(i)
    }
    return ans;
}
```

### Brute Force Alternative: O(n log n)

```csharp
// Count bits for each number individually
public int[] CountBits(int n)
{
    int[] result = new int[n + 1];
    for (int i = 0; i <= n; i++)
    {
        int count = 0;
        int num = i;
        while (num > 0)
        {
            count += num & 1;
            num >>= 1;
        }
        result[i] = count;
    }
    return result;
}
```

**Why DP is Better:** O(n) vs O(n log n) by reusing previous results instead of recalculating for each number.

## Strategy 4: Bit Manipulation Arithmetic

**Bit Manipulation Arithmetic** implements mathematical operations using only bitwise operators. Addition without + operator uses XOR for sum (without carry) and AND + left shift for carry. This technique reveals the underlying hardware implementation and enables solving problems with restricted operators. Understanding that addition is fundamentally XOR + carry propagation is key.

### When to use it

- Implement arithmetic without +, -, *, / operators
- Understand hardware-level operations
- Problems explicitly forbidding arithmetic operators
- Educational understanding of binary arithmetic

### Time Complexity: O(1) for fixed-size integers | Space Complexity: O(1)

### Problem: Sum of Two Integers

Calculate sum of two integers without using + operator.

**Implementation:**

```csharp
public class Solution
{
    public int GetSum(int a, int b)
    {
        int carry = 0, ans = 0;

        for (int i = 0; i < 32; ++i)
        {
            int x = (a >> i & 1); // Get i-th bit of a
            int y = (b >> i & 1); // Get i-th bit of b

            if (carry != 0)
            {
                if (x == y)
                {
                    ans |= 1 << i; // Set bit in result
                    if (x == 0 && y == 0) carry = 0;
                }
            }
            else
            {
                if (x != y) ans |= 1 << i;
                if (x == 1 && y == 1) carry = 1;
            }
        }

        return ans;
    }
}
```

**Key Insight:** Addition at bit level: XOR gives sum without carry, AND gives carry positions. Propagate carry left and repeat until no carry remains.

**How Binary Addition Works:**
```
  0101 (5)
+ 0011 (3)
-------
Step 1: XOR (sum without carry)
  0101 ^ 0011 = 0110
Step 2: AND << 1 (carry)
  (0101 & 0011) << 1 = 0010
Step 3: Repeat until carry = 0
  0110 + 0010 = 1000 (8)
```

**Simpler iterative approach:**
```csharp
public int GetSum(int a, int b)
{
    while (b != 0)
    {
        int carry = (a & b) << 1; // Calculate carry
        a = a ^ b; // Sum without carry
        b = carry; // New b is the carry
    }
    return a;
}
```

### Brute Force Alternative: O(1)

```csharp
// Just use + operator (not allowed in problem)
public int GetSum(int a, int b)
{
    return a + b; // Too easy, defeats the purpose
}
```

**Why Bit Manipulation is Required:** Problem explicitly forbids arithmetic operators to test understanding of binary arithmetic fundamentals.

## Strategy Selection Guide

| Problem Type | Strategy | Time | Space |
|-------------|----------|------|-------|
| Count 1 bits | Brian Kernighan | O(k) | O(1) |
| Find missing number | XOR Properties | O(n) | O(1) |
| Count bits 0 to n | DP with Bits | O(n) | O(n) |
| Sum without + | Bit Manipulation | O(1) | O(1) |
| Power of 2 check | n & (n-1) == 0 | O(1) | O(1) |
| Swap variables | XOR swap | O(1) | O(1) |
| Reverse bits | Bit manipulation | O(1) | O(1) |
| Single number | XOR all elements | O(n) | O(1) |

## Pattern Recognition Tips

1. **"Count 1 bits"** → Brian Kernighan's Algorithm (n & (n-1))
2. **"Find unique/missing"** → XOR properties (self-cancellation)
3. **"Bits from 0 to n"** → Dynamic Programming
4. **"Without +/- operator"** → Bit manipulation arithmetic
5. **"Power of 2"** → Check if n & (n-1) == 0
6. **"Swap without temp"** → XOR swap: a^=b, b^=a, a^=b
7. **"Isolate rightmost bit"** → n & -n

## Essential Bit Operations

### Basic Bitwise Operators
```csharp
// AND: Both bits must be 1
5 & 3 → 0101 & 0011 = 0001 (1)

// OR: At least one bit must be 1
5 | 3 → 0101 | 0011 = 0111 (7)

// XOR: Bits must be different
5 ^ 3 → 0101 ^ 0011 = 0110 (6)

// NOT: Flip all bits
~5 → ~0101 = 1010 (in 4-bit)

// Left Shift: Multiply by 2^n
5 << 2 → 0101 << 2 = 10100 (20)

// Right Shift: Divide by 2^n
5 >> 1 → 0101 >> 1 = 0010 (2)
```

### Common Bit Tricks
```csharp
// Check if i-th bit is set
bool IsBitSet(int n, int i) => (n & (1 << i)) != 0;

// Set i-th bit
int SetBit(int n, int i) => n | (1 << i);

// Clear i-th bit
int ClearBit(int n, int i) => n & ~(1 << i);

// Toggle i-th bit
int ToggleBit(int n, int i) => n ^ (1 << i);

// Get rightmost set bit
int RightmostSetBit(int n) => n & -n;

// Clear rightmost set bit
int ClearRightmostSetBit(int n) => n & (n - 1);

// Check if power of 2
bool IsPowerOfTwo(int n) => n > 0 && (n & (n - 1)) == 0;

// XOR swap (no temp variable)
void XorSwap(ref int a, ref int b)
{
    a ^= b;
    b ^= a;
    a ^= b;
}
```

### Counting and Checking
```csharp
// Count trailing zeros
int CountTrailingZeros(int n) => 
    n == 0 ? 32 : (int)Math.Log2(n & -n);

// Check if two integers have opposite signs
bool OppositeSigns(int x, int y) => (x ^ y) < 0;

// Absolute value without branching
int Abs(int n)
{
    int mask = n >> 31;
    return (n ^ mask) - mask;
}
```

## Conclusion

Mastering bit manipulation is essential for efficient low-level programming and understanding computer architecture. The key is to:

1. **Understand binary representation** - How numbers are stored as bits
2. **Learn bitwise operators** - AND, OR, XOR, NOT, shifts
3. **Recognize patterns** - XOR cancellation, power of 2, Brian Kernighan
4. **Practice bit tricks** - Common operations become intuitive with practice

Bit manipulation often provides the most elegant and efficient solutions. With practice, you'll recognize when bitwise operations can replace more complex algorithms. 🔢

## References

_LeetCode. (2024). Bit Manipulation Problems. https://leetcode.com/tag/bit-manipulation/_

_Warren, H. S. (2012). Hacker's Delight (2nd ed.). Addison-Wesley._

_Cormen, T. H., Leiserson, C. E., Rivest, R. L. & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press._
