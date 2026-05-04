---
title: Understanding Compilers Through an Algebraic Expression Compiler
description: Learn compiler design fundamentals by building an algebraic expression compiler in F#. Understand tokenization, parsing, AST construction, semantic analysis, and code generation.
Author: Jonas Lara
date: 2026-05-03 00:00:00 +0000
categories: [Compiler Design, F#, Programming Languages]
tags: [fsharp, compiler, parsing, ast, code generation, language design]
image:
  path: /assets/img/post/understanding-compilers-through-algebra/algebra.jpg
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post-understanding-compilers-through-algebra/algebra.jpg
  alt: Algebraic expression with variables and numbers
---

# Understanding Compilers Through an Algebraic Expression Compiler

Building a compiler is one of the best ways to understand how programming languages work under the hood. Instead of diving into complex language features, we'll learn compiler design principles by creating a focused, practical project: an **Algebraic Tiny Compiler (ATC)** in F#.

This compiler transforms algebraic expressions into executable code, demonstrating every major compiler phase: lexing, parsing, semantic analysis, and code generation.

> **Disclaimer**: The code shown in this post is simplified for clarity and educational purposes. The [complete repository](https://github.com/jonas1ara/algebraic-tiny-compiler) contains more robust implementations with additional error handling, edge cases, and optimizations.

## Why F# is Ideal for Compilers 

F# has built-in features that make compiler writing elegant and safe:

### 1. **Discriminated Unions (DU) = Perfect AST Representation**

Instead of messy class hierarchies with nullable fields, F# discriminated unions provide type-safe algebraic data types:

```fsharp
// F# way - type-safe, exhaustive pattern matching
type Expr = 
    | Number of float
    | Variable of string
    | BinOp of Expr * string * Expr
    | Power of Expr * float

// Every case is handled by the compiler—impossible to miss one!
```

**Compared to Java/C#:**
```csharp
// Mess: nullable fields, unsafe type casting
abstract class Expr { }
class Number : Expr { public double Value; }
class Variable : Expr { public string Name; }
class BinOp : Expr { public Expr Left; public Expr Right; public string Op; }

// You can accidentally forget to handle a case—the compiler won't catch it
```

### 2. **Pattern Matching = Simpler Parsing & Semantic Analysis**

Pattern matching makes recursive tree processing intuitive:

```fsharp
// Checking if an expression is linear in x
let rec isLinear expr var =
    match expr with
    | Number _ -> true
    | Variable v -> v = var
    | Power(_, n) when n = 1.0 -> true  // Only degree 1
    | BinOp(left, "+", right) -> isLinear left var && isLinear right var
    | BinOp(left, "-", right) -> isLinear left var && isLinear right var
    | _ -> false
```

### 3. **Immutability = Safer Compiler Pipelines**

Each compilation phase takes immutable input and produces immutable output. No surprising mutations:

```fsharp
// Pure functions—input never changes
let tokenize (input: string) : Token list = ...
let parse (tokens: Token list) : Expr = ...
let analyze (expr: Expr) : Polynomial = ...

// Pipeline: input → tokens → AST → polynomial → solutions
// Each step independent, testable, and composable
```

### 4. **Pure Functions = Predictable Compiler Behavior**

No hidden side effects or global state. Each function's output depends only on its inputs:

```fsharp
// This function always returns the same result for the same input
// Easy to test, cache, parallelize
let solveQuadratic a b c : float list =
    let discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0 then []
    else [(-b + sqrt discriminant) / (2.0 * a)]

// Safe to run in parallel, memoize, or test in isolation
```

---

## Why Learn Compilers?

Understanding compilers is valuable for:

- **Language Design**: Creating domain-specific languages (DSLs)
- **Performance Optimization**: Understanding how code gets compiled
- **Debugging**: Better insight into what's happening under the hood
- **Functional Programming**: Mastering fundamental CS concepts
- **Career Growth**: Advanced technical skills employers value

> **The Plan**: We'll build a compiler that can parse, simplify, solve, and generate code for algebraic expressions—all while learning industry-standard compiler techniques.

## The Compiler Pipeline

Every compiler follows a similar architecture:

```
Input: "2x^2 + 3x - 5 = 0"
  ↓
[TOKENIZATION] → Tokens: [Num(2), Var(x), Caret, Num(2), ...]
  ↓
[PARSING] → AST: Equation(Left: Polynomial(...), Right: Num(0))
  ↓
[SEMANTIC ANALYSIS] → Simplified: 2x² + 3x - 5 = 0
  ↓
[SOLVING] → Solutions: x = 1.0, x = -2.5
  ↓
[CODE GENERATION] → Assembly-like output
```

Let's explore each phase in detail.

## Phase 1: Tokenization (Lexical Analysis)

### The Problem

When the compiler receives raw text like `"2x^2 + 3x - 5 = 0"`, it's just a string of characters. We need to break it into meaningful units (tokens) that we can process.

**Without tokenization**, you'd have to manually scan each character:
```
"2x^2 + 3x - 5 = 0"
 ^
 Is this '2'? Is there an 'x' after? What about '^'?
```

**With tokenization**, we get clean tokens:
```
Token {Type = Number, Value = "2"}
Token {Type = Variable, Value = "x"}
Token {Type = Power, Value = "^"}
Token {Type = Number, Value = "2"}
... and so on
```

### Implementation in F#

```fsharp
// Token definition
type Token = {
    Type: TokenType
    Value: string
}

and TokenType =
    | Number
    | Variable
    | Plus
    | Minus
    | Multiply
    | Divide
    | Power
    | LeftParen
    | RightParen
    | Equals
    | End

// Simple tokenizer
let tokenize (input: string) : Token list =
    let rec helper (chars: char list) (tokens: Token list) =
        match chars with
        | [] -> tokens @ [{ Type = End; Value = "" }]
        
        | ' ' :: rest -> helper rest tokens  // Skip whitespace
        
        | c :: rest when Char.IsDigit(c) ->
            let numStr, remaining = takeWhile Char.IsDigit (c :: rest)
            let token = { Type = Number; Value = numStr }
            helper remaining (tokens @ [token])
        
        | c :: rest when Char.IsLetter(c) ->
            let varStr, remaining = takeWhile Char.IsLetterOrDigit (c :: rest)
            let token = { Type = Variable; Value = varStr }
            helper remaining (tokens @ [token])
        
        | '+' :: rest -> helper rest (tokens @ [{ Type = Plus; Value = "+" }])
        | '-' :: rest -> helper rest (tokens @ [{ Type = Minus; Value = "-" }])
        | '*' :: rest -> helper rest (tokens @ [{ Type = Multiply; Value = "*" }])
        | '/' :: rest -> helper rest (tokens @ [{ Type = Divide; Value = "/" }])
        | '^' :: rest -> helper rest (tokens @ [{ Type = Power; Value = "^" }])
        | '(' :: rest -> helper rest (tokens @ [{ Type = LeftParen; Value = "(" }])
        | ')' :: rest -> helper rest (tokens @ [{ Type = RightParen; Value = ")" }])
        | '=' :: rest -> helper rest (tokens @ [{ Type = Equals; Value = "=" }])
        | c :: _ -> failwith $"Unknown character: {c}"
    
    helper (input.ToCharArray() |> Array.toList) []
```

### Key Insights

1. **Regular scanning**: Process input left-to-right, extracting tokens one at a time
2. **Whitespace handling**: Skip insignificant characters
3. **Multi-character tokens**: Numbers like `123` and variables like `xyz` need grouping
4. **End marker**: Add a sentinel token to signal completion

### Tokenizer Limitations (Intentional Simplifications)

The tokenizer shown above is intentionally simplified for clarity. Real tokenizers handle more complex cases:

- ❌ **Decimal numbers**: `3.14` not supported (only integers)
- ❌ **Negative literals**: `-5` is parsed as `Minus` + `Num(5)`, not `Num(-5)`
- ❌ **Scientific notation**: `1e-3` not supported
- ❌ **Multi-character operators**: `**` (power) instead of `^` not supported
- ❌ **Comments**: `# this is a comment` not supported
- ✅ **What works**: Single-char operators, positive integers, single-letter variables

**Why this matters**: In production compilers, tokenizers handle these cases with regex or finite-state automata (FSA). Our recursive approach works for simple cases but would become unwieldy with these extensions.

**Challenge**: Extend the tokenizer to support decimal numbers! (Hint: Track whether you've seen a dot yet)

## Phase 2: Parsing (Syntax Analysis)

### The Problem

Tokens are still flat—we need to understand their **structure and precedence**. Consider:

```
"2 + 3 * 4"

Tokens: [2, +, 3, *, 4]

But should it be: (2 + 3) * 4 = 20?   ❌ WRONG
Or should it be:  2 + (3 * 4) = 14?   ✅ CORRECT (operator precedence!)
```

### Recursive Descent Parsing

**Recursive descent** is a top-down parsing technique that respects operator precedence by structuring the grammar hierarchically.

The key principle: **Higher precedence = deeper in the recursion tree**

#### Understanding ParserState: Tracking Progress Through Tokens

Before we look at the parser code, let's understand the `ParserState` type that tracks our position:

```fsharp
type ParserState = {
    Tokens: Token list
    Index: int              // Current position (0, 1, 2, ...)
    CurrentToken: Token    // Tokens[Index]
}

// Helper to move forward
let advance state =
    let nextIndex = state.Index + 1
    {
        Tokens = state.Tokens
        Index = nextIndex
        CurrentToken = if nextIndex < state.Tokens.Length then state.Tokens[nextIndex] else { Type = End; Value = "" }
    }
```

**Example walkthrough**: For tokens `[Num(2), Plus, Num(3)]`

```
Initial State: Index=0, CurrentToken=Num(2)
    Tokens: [Num(2), Plus, Num(3), End]
            ^Index
            
After advance(): Index=1, CurrentToken=Plus
    Tokens: [Num(2), Plus, Num(3), End]
                    ^Index
                    
After advance(): Index=2, CurrentToken=Num(3)
    Tokens: [Num(2), Plus, Num(3), End]
                         ^Index
```

Each parser function **reads** from the current token and **returns** the updated state.

#### Grammar and Recursive Structure

```fsharp
// Expression grammar:
// expr       := term (('+' | '-') term)*
// term       := factor (('*' | '/') factor)*
// factor     := primary ('^' primary)*
// primary    := number | variable | '(' expr ')'

let rec parseExpression state =
    let left, state = parseTerm state
    parseExpressionHelper left state

and parseExpressionHelper left state =
    match state.CurrentToken.Type with
    | Plus | Minus as op ->
        let right, state = parseTerm (advance state)
        let expr = Binop(opToString op, left, right)
        parseExpressionHelper expr state
    | _ -> (left, state)

and parseTerm state =
    let left, state = parseFactor state
    parseTermHelper left state

and parseTermHelper left state =
    match state.CurrentToken.Type with
    | Multiply | Divide as op ->
        let right, state = parseFactor (advance state)
        let expr = Binop(opToString op, left, right)
        parseTermHelper expr state
    | _ -> (left, state)

and parseFactor state =
    let left, state = parsePrimary state
    parsePowerHelper left state

and parsePowerHelper left state =
    match state.CurrentToken.Type with
    | Power ->
        let right, state = parsePrimary (advance state)
        let expr = Power(left, exprToInt right)
        (expr, state)
    | _ -> (left, state)

and parsePrimary state =
    match state.CurrentToken.Type with
    | Number -> (Num (float state.CurrentToken.Value), advance state)
    | Variable -> (Var state.CurrentToken.Value, advance state)
    | LeftParen ->
        let expr, state = parseExpression (advance state)
        (expr, advance state)  // Skip closing paren
    | _ -> failwith "Expected number, variable, or ("
```

### The AST (Abstract Syntax Tree)

After parsing, we have a tree structure:

```fsharp
type Expr =
    | Num of float
    | Var of string
    | Binop of string * Expr * Expr
    | Power of Expr * int
    | Paren of Expr
    | UnaryOp of string * Expr
```

For `"2 * x + 3"`, the AST is:

```
        Binop(+)
        /        \
    Binop(*)      Num(3)
    /      \
  Num(2)   Var(x)
```

### Key Insights

1. **Precedence through recursion**: Lower precedence operations at the top
2. **Left associativity**: `a - b - c` becomes `(a - b) - c`
3. **Modularity**: Each function handles one precedence level
4. **Error detection**: Invalid syntax caught during parsing

## Phase 3: Semantic Analysis & Simplification

### The Problem

The parser gives us the correct structure, but:
- `2x` should be treated as `2 * x`
- `x + x` should simplify to `2x`
- `(x + 1)(x - 1)` should expand to `x² - 1`

This is **semantic analysis**: understanding what the parsed code *means*.

### Representation: Polynomials

Instead of keeping arbitrary expression trees, we convert to a **canonical polynomial form**. This normalization makes solving equations much simpler.

#### Why Canonical Form?

Different expression trees can represent the same polynomial:

```
"2x + 3x" 
  ↓ 
AST: Add(Mul(2, Var(x)), Mul(3, Var(x)))

"3x + 2x"
  ↓
AST: Add(Mul(3, Var(x)), Mul(2, Var(x)))

"5x"
  ↓
AST: Mul(5, Var(x))

All three represent the SAME mathematical expression!
```

**Canonical form** collapses these into one representation:

```
Before (AST): 
  - Different tree structures for equivalent expressions
  - Hard to recognize "3x + 2x = 5x"
  - Solving logic must handle all variants

After (Canonical Polynomial):
  - Single form: [Term(5, "x", 1)]
  - Easy to combine like terms
  - Solving logic is simple and predictable
```

#### Polynomial Representation

```fsharp
type Term = {
    Coefficient: float
    Variable: string
    Power: int
}

type Polynomial = Term list
// Example: 2x² + 3x - 5
// becomes: [Term(2, "x", 2); Term(3, "x", 1); Term(-5, "", 0)]
```

**Why this representation?**
- Each term is atomic (can't be simplified further)
- Grouping by `(Variable, Power)` is O(n)
- Solving equations only needs these three pieces

### Expansion Algorithm

```fsharp
let rec expandExpression (expr: Expr) : Polynomial option =
    match expr with
    | Num n ->
        Some [{ Coefficient = n; Variable = ""; Power = 0 }]
    
    | Var v ->
        Some [{ Coefficient = 1.0; Variable = v; Power = 1 }]
    
    | Binop("+", left, right) ->
        match expandExpression left, expandExpression right with
        | Some l, Some r -> Some (combineLikeTerms (l @ r))
        | _ -> None
    
    | Binop("*", left, right) ->
        match expandExpression left, expandExpression right with
        | Some l, Some r -> Some (multiplyPolynomials l r)
        | _ -> None
    
    | Power(base', exp) ->
        // Handle x^2, (2x)^3, etc.
        // ...
    
    | _ -> None
```

**Note**: `multiplyPolynomials` multiplies each term in the first polynomial by each term in the second. It's implemented in the repository but omitted here for brevity. 

### Combining Like Terms

```fsharp
let combineLikeTerms (terms: Term list) : Term list =
    terms
    |> List.groupBy (fun t -> (t.Variable, t.Power))
    |> List.map (fun ((var, pow), group) ->
        let coeff = group |> List.sumBy (fun t -> t.Coefficient)
        { Coefficient = coeff; Variable = var; Power = pow }
    )
    |> List.filter (fun t -> t.Coefficient <> 0.0)
```

### Example: Simplifying `(x + 1)²`

```
Input: Power(Binop("+", Var "x", Num 1), 2)

Expand: (x + 1) * (x + 1)
      = x*x + x*1 + 1*x + 1*1
      = x² + x + x + 1
      = x² + 2x + 1

Output: [Term(1, "x", 2); Term(2, "x", 1); Term(1, "", 0)]
```

### Key Insights

1. **Canonical form**: Multiple expressions may represent the same polynomial
2. **Automatic simplification**: Like terms combined through grouping
3. **Type safety**: Polynomial operations checked by the F# compiler
4. **Composability**: Each function is a pure, testable unit

## Phase 4: Equation Solving

### Types of Equations We Solve

```fsharp
// Linear: ax + b = 0 → x = -b/a
// Quadratic: ax² + bx + c = 0 → x = (-b ± √(b² - 4ac)) / 2a
```

### Linear Solver

```fsharp
let solveLinear (a: float) (b: float) : float list =
    if a = 0.0 then
        if b = 0.0 then
            []  // 0 = 0 (infinite solutions - we return empty)
        else
            []  // b ≠ 0 (no solution)
    else
        [-b / a]
```

### Quadratic Solver

```fsharp
let solveQuadratic (a: float) (b: float) (c: float) : float list =
    if a = 0.0 then
        solveLinear b c
    else
        let discriminant = b * b - 4.0 * a * c
        
        if discriminant < 0.0 then
            []  // Complex roots (not real)
        elif discriminant = 0.0 then
            [-b / (2.0 * a)]  // One solution
        else
            let sqrt_d = System.Math.Sqrt(discriminant)
            [
                (-b + sqrt_d) / (2.0 * a)
                (-b - sqrt_d) / (2.0 * a)
            ]
```

### Example: Solving `2x² + 3x - 5 = 0`

```
a = 2, b = 3, c = -5

discriminant = 3² - 4(2)(-5)
            = 9 + 40
            = 49

√49 = 7

x₁ = (-3 + 7) / 4 = 4/4 = 1
x₂ = (-3 - 7) / 4 = -10/4 = -2.5

Solutions: [1.0, -2.5]
```

### Key Insights

1. **Mathematical correctness**: Accurate handling of edge cases
2. **Numerical stability**: Careful with floating-point arithmetic
3. **Completeness**: Handle all solution cases (none, one, two, infinite)

## Phase 5: Code Generation

### What Is Code Generation?

Instead of just evaluating expressions, we generate a lower-level representation—like assembly code or an intermediate language.

### Example: `2x + 3`

```fsharp
// Generated code:
[
    "LOAD 2";              // Load 2 into register 0
    "LOAD_VAR x";          // Load x into register 1
    "MUL R0 R1";           // R0 = 2 * x
    "LOAD 3";              // Load 3 into register 2
    "ADD R0 R2";           // R0 = 2x + 3
    "RETURN R0";           // Return result
]
```

### Register Allocation

```fsharp
let generateCode (expr: Expr) : string list =
    let mutable regCount = 0
    
    let rec gen (expr: Expr) (regNum: int) : (string list * int) =
        match expr with
        | Num n ->
            ([$"LOAD {n}"], regNum + 1)
        
        | Var v ->
            ([$"LOAD_VAR {v}"], regNum + 1)
        
        | Binop(op, left, right) ->
            let leftCode, nextReg = gen left regNum
            let rightCode, finalReg = gen right nextReg
            let opCode = match op with
                        | "+" -> "ADD"
                        | "-" -> "SUB"
                        | "*" -> "MUL"
                        | "/" -> "DIV"
                        | _ -> "UNKNOWN"
            
            let code = leftCode @ rightCode @
                      [$"{opCode} R{nextReg-2} R{nextReg-1}";
                       $"MOVE R{nextReg-2} R{finalReg-1}"]
            
            (code, finalReg)
        
        | _ -> ([], regNum)
    
    let code, _ = gen expr 0
    code @ ["RETURN R0"]
```

### Why Code Generation?

1. **Optimization**: Generate efficient code
2. **Portability**: Generate different code for different targets
3. **Interpretation**: Execute generated code
4. **Learning**: Understand instruction sets and register allocation

### Key Insights

1. **Register allocation**: Minimal, efficient register usage
2. **Instruction sequence**: Correct order for dependencies
3. **Modularity**: Reusable code generation components

## The Complete Pipeline in Action

### Example: `x^2 - 3x + 2 = 0`

```
Step 1: TOKENIZATION
Input:  "x^2 - 3x + 2 = 0"
Output: [Var(x), Power(^), Num(2), Minus, Num(3), Var(x), Plus, Num(2), Equals, Num(0)]

Step 2: PARSING
Output: Equation(
    Left: Binop(+, Binop(-, Power(Var x, 2), Binop(*, Num 3, Var x)), Num 2),
    Right: Num 0
)

Step 3: SIMPLIFICATION
Output: Polynomial([
    Term(1, "x", 2),
    Term(-3, "x", 1),
    Term(2, "", 0)
])

Step 4: SOLVING
Using quadratic formula with a=1, b=-3, c=2:
discriminant = 9 - 8 = 1
x = (3 ± 1) / 2 = [2.0, 1.0]

Step 5: CODE GENERATION
[
    "LOAD 1",
    "LOAD_VAR x",
    "POW R0 R1 2",
    "LOAD 3",
    "LOAD_VAR x",
    "MUL R2 R3",
    "SUB R0 R2",
    "LOAD 2",
    "ADD R0 R4",
    "RETURN R0"
]
```


## What's Next?

You've mastered the fundamentals of compiler design. Now it's time to take the next step:

**[Part 2: Building a Better Algebraic Compiler - Advanced Techniques]({{ "/posts/building-better-algebraic-compiler/" | relative_url }})** covers:

- ✅ **Error Recovery**: How real compilers handle multiple errors
- ✅ **Optimization Passes**: Making generated code faster
- ✅ **Compilation Models**: JIT vs Ahead-of-Time compilation
- ✅ **Real-World Projects**: Studying production F# compiler source code

In Part 2, we'll also explore:
- Incremental parsing for IDE responsiveness
- Code generation strategies
- Performance profiling techniques
- How language servers integrate with compilers

---

*Ready for the challenge? Continue to [Part 2](2026-05-04-building-better-algebraic-compiler.md)* 🚀
