---
title: "Another Reflexion about Rust from a C# Developer"
description: "Closing the Glance Rust series with a look back: how learning Rust's ownership model, enums, and compiler discipline quietly made me a better C# developer — and why the industry stopped treating Rust as speculative."
Author: Jonas Lara
date: 2026-07-22 00:00:00 +0000
categories: [Software Engineering, Rust, .NET]
tags: [rust, dotnet, csharp, memory-safety, language-design, glance]
image:
  path: /assets/img/post/another-reflexion-about-rust/RustCsharp.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/another-reflexion-about-rust/RustCsharp.png
  alt: Rust and C# side by side
---

# Another Reflexion about Rust from a C# Developer

### Closing a loop

Over the last few posts I've written about building Glance's backend in Rust — the borrow checker fights, the `async` runtime decisions, the moments where the compiler rejected code that would have compiled (and quietly misbehaved) in any other language. That series is done now. Glance works, it's fast, and it didn't crash once in a way that traced back to a null reference or a data race.

But somewhere in that process, something I didn't plan for happened: I became a better C# developer. Not because I switched languages — I still write C# every day — but because Rust rewired how I *think* about the code I already knew how to write.

This is the reflection post. No new crate, no benchmark, no Glance feature. Just what's left once the dust settles.

### The compiler stopped being an obstacle and became a second opinion

In C#, I used to treat the compiler as a gate: does it build, yes/no. Rust's compiler doesn't just gate — it *argues*. It asks who owns this value, how long it lives, whether two things can mutate it at once. At first that felt hostile. After a few months, I noticed I'd started asking myself those same questions *before* writing the C# equivalent, instead of finding out at 2am from a production exception.

That shows up in small, concrete ways:

- I reach for `readonly`, `record`, and `record struct` by default now, and only relax to a mutable class when I have a real reason — not the other way around.
- I actually read the nullable reference type warnings instead of sprinkling `!` to make them go away.
- When I return errors, I lean toward an explicit result type (a small `Result<T, TError>`-style struct) for expected failure paths, and reserve exceptions for things that are genuinely exceptional — a very Rust-shaped distinction that C#'s exception-happy culture doesn't encourage on its own.
- I've started noticing allocations I would have ignored before — where a `Span<T>` or pooling an array actually matters, instead of trusting the GC to make it a non-problem.

None of this makes C# into Rust. It makes the C# I write more deliberate.

### Beyond stack vs. heap: choosing struct over class (and record) wisely

Value types are close to free and vanish the instant a function returns; heap allocations cost something real, and someone always has to account for it later. Rust makes you learn that distinction whether you want to or not. What actually changed how I write C# day to day, though, is what that awareness turned into once it stopped being trivia.

I used to default to `class` for almost everything, and to `record` once C# 9 made immutable types convenient — not out of any real reasoning, just habit. Rust doesn't let you coast like that: the path of least resistance is a value type, and reference semantics (`Box`, `Rc`, `Arc`) are things you reach for deliberately, when you actually have a reason.

That pushed me to go back and actually read .NET's own framework design guidelines on struct vs. class — advice that's been sitting in the docs for years, unused, because nothing had forced me to care. The short version: a `struct` earns its keep when it's small (a rough rule of thumb: under ~16 bytes), short-lived, and represents a single value rather than an identity — a 2D point, a small ID wrapper, a descriptor built and consumed inside one method. Past that size, copying it around on every call starts costing more than the heap allocation it was supposed to avoid.

The same question now applies to `record` vs. `record struct`: am I modeling an immutable *value*, or an immutable *thing with identity*? A `record class` still costs a heap allocation and a GC entry per instance; a `record struct` doesn't. Neither answer is universally correct — but I didn't used to ask the question at all, and now I do, for the same reason I check whether a Rust value needs a `Box`: the cost is real whether or not I bother to notice it.

### Enums taught me what C# was missing, and it's finally arriving

Back in 2024, when I was still just learning Rust's syntax, `enum` was the one feature that made me genuinely jealous of Rust from the C# side of the fence. Not ownership, not the borrow checker — `enum`. It was the first time a language feature made me wish I could bring something *back* to C# instead of the other way around.

If there's one Rust feature that rewired how I think about modeling data, it's `enum`. Not the C-style enum C# has had forever — a Rust `enum` is a real sum type. Each variant can carry its own data:

```rust
enum PaymentResult {
    Success { transaction_id: String },
    Declined { reason: String },
    NetworkError,
}
```

And `match` forces you to handle every variant. Miss one, and the compiler stops you — not a code reviewer, not a runtime exception three months later, the compiler, at the moment you wrote the gap.

F# developers have had this for two decades under the name *discriminated unions*, and it's the same idea: a closed set of named cases, each optionally carrying data, matched exhaustively. If you know me from X or LinkedIn, you know I'm around there as an F# developer as much as anything else — I already knew this had been sitting in F# for two decades before I ever touched Rust. But let's be honest: the internet didn't lose its mind over F#. It lost its mind over Rust. No shade to F#, which I still love — it's just that one of these two languages got the entire industry to notice a feature the other one had first. Once you've modeled a domain this way — a payment result, an HTTP response, a parser AST — going back to "a class with a nullable field for each possible outcome plus a comment saying which ones are mutually exclusive" feels like modeling with duct tape.

C# never had this natively. We simulated it: class hierarchies with pattern matching, `switch` expressions over `is` patterns, libraries like `OneOf` filling the gap. All workable, none of it exhaustive at compile time the way Rust and F# are.

That's finally changing. **C# 15, shipping with .NET 11**, is adding real `union` types — currently in preview (`.NET 11 Preview 2`, behind `<LangVersion>preview</LangVersion>`), with general availability expected around November 2026:

```csharp
union Pet(Cat, Dog, Bird) { }
```

with pattern matching that the compiler can actually verify is exhaustive. It's taken over a decade of asking on the C# language design repo to get here.

One honest technical caveat, because I don't want to oversell it: the C# proposal is explicit that this is a union of *types*, not a tagged/discriminated union in the strict sense Rust and F# use — there's no named case carrying its own data the way `Success { transaction_id }` does, it lowers to a struct with a single `object`-typed `Value` and a generated constructor per case type. You *can* express discriminated-union-shaped code with it, it's just a different mechanism underneath. Still — after a decade of workarounds, C# developers are finally getting the exhaustiveness guarantee, and that's the part that actually changes how you model a domain. I know exactly why I want it now. I didn't, before Rust taught me the shape of the problem.

### The Rust era is here

For years, Rust hype had a specific shape: enthusiasm running well ahead of adoption. Every year was supposedly "the year of Rust," every conference had a talk about the future it promised, and it was fair to read most of it as hope outrunning evidence. That gap has been closing, and — for once — closing in the direction everyone hoped for. The promises are turning into shipped patches, published vulnerability numbers, and kernel commits, not just keynote slides.

I used to treat "should I learn Rust" as a bet on a possible future. It isn't anymore. Two data points made that obvious to me.

The first time Rust actually made noise across the industry was because of this.

The first is Mark Russinovich, Azure's CTO, who has been saying for years — first in a 2019 BlueHat IL talk, then publicly and repeatedly since 2022 — that roughly **70% of the CVEs Microsoft has had to patch across its own software trace back to the same root cause: memory safety bugs in C and C++**. Not seventy different problems. One problem, wearing seventy different masks, for decades. His conclusion wasn't "be more careful." It was: stop starting new projects in languages where this class of bug is even possible, and use Rust instead.

That wasn't just a keynote line Microsoft quietly forgot about. It turned into a training book and a pile of shipping software. Microsoft has published an entire [Rust-for-C#-developers training book](https://microsoft.github.io/RustTraining/csharp-book/ch00-introduction.html) that maps ownership vs. GC, traits vs. interfaces, `Result` vs. exceptions, Cargo vs. NuGet — concept by concept, in terms a C# developer already understands. And Windows itself is quietly being rebuilt around exactly what Russinovich said back then:

- [**windows-rs**](https://github.com/microsoft/windows-rs) — the official crate for calling the entire Win32/WinRT API surface from Rust.
- [**windows-reactor**](https://github.com/microsoft/microsoft-ui-reactor) — a React-inspired, declarative UI framework for building native WinUI 3 apps in pure Rust.
- [**windows-drivers-rs**](https://github.com/microsoft/windows-drivers-rs) — a real path to writing WDM/KMDF/UMDF drivers in Rust; the Surface team has already shipped drivers built on it.
- [**edit**](https://github.com/microsoft/edit) — the new terminal text editor Microsoft ships with Windows, written in Rust, under 250KB.
- [**sudo for Windows**](https://github.com/microsoft/sudo) — yes, `sudo` on Windows now exists, and it's written in Rust.
- [**Coreutils for Windows**](https://learn.microsoft.com/en-us/windows/core-utils/overview) — `ls`, `cp`, `grep`, `find`, and dozens more Unix staples, running natively on Windows, built on the Rust-based `uutils/coreutils` project.

And it's not just the client side of Windows — Azure itself moved the same direction. [**Azure Boost**](https://learn.microsoft.com/en-us/azure/azure-boost/overview), the hardware-offload layer sitting under Azure's VMs, made Rust the primary language for all new code on the system. [**OpenVMM**](https://opensource.microsoft.com/blog/2024/11/07/introducing-hyperlight-virtual-machine-based-security-for-functions-at-scale/) and Hyperlight are Rust-based virtualization projects built to isolate workloads at the hypervisor level. And in May 2026, the [**Azure SDK for Rust**](https://devblogs.microsoft.com/azure-sdk/from-beta-to-stable-announcing-the-azure-sdk-for-rust-ga/) hit general availability — stable 1.0 crates for Identity, Key Vault, and Storage, meaning Rust is no longer just how Microsoft builds Azure, it's now a first-class, supported way to *call* Azure too.

That's not a research bet. That's kernel-adjacent, developer-facing, cloud-infrastructure-facing, ship-to-a-billion-machines commitment — the same CTO who pointed at the 70% number years ago, followed all the way through to drivers, hypervisors, and terminal tools shipping in Rust today.

But let's be honest about who's saying it: a lot of developers distrust anything that comes out of Redmond on principle, no matter how good the data — or the follow-through — behind it is. So if there's a moment that actually won over that crowd — the one that doesn't take Microsoft's word for anything — it's probably not that one. It's this one.

The second is what just happened in the Linux kernel. For years, Rust support in Linux was labeled *experimental* — a hedge, a maybe. In December 2025, at the Kernel Maintainer Summit in Tokyo, the Rust-for-Linux team submitted a patch simply titled "conclude the Rust experiment." Not because it failed — because it succeeded enough that calling it an experiment stopped making sense. Torvalds' own position settled it further: no individual maintainer gets to veto Rust in their subsystem on philosophical grounds anymore. The experiment is over. Rust stays.

Put those two together and the message is hard to miss: this isn't a language a curious developer bets on anymore, hoping the ecosystem catches up. The ecosystem already caught up. The people running the two operating systems most of the world's software depends on both looked at the same evidence and landed in the same place.

### The point of learning a language you might not switch to

I'm not going to stop writing C#. That was never the plan, and to be clear, I'm not telling you to drop C# to go learn Rust either. But I'd tell any C# developer curious about Rust the same thing I'd have wanted to hear before I started: you're not learning it to leave C# behind, and you're not betting on a language that might not pan out. That question is already answered — by Microsoft's own numbers, by the kernel that runs most of the internet. You're learning it so that the next time you write a class, define an error path, or reach for `new`, there's a quieter, stricter voice in your head asking the question Rust never lets you skip.

What actually changes isn't any single feature — it's that Rust forces you to think through your logic carefully enough to satisfy a compiler that refuses to take your word for it, and that habit doesn't turn off when you close the editor. Immutability by default, nullable reference types treated as a real contract instead of a warning to silence, a `Result`-shaped return instead of an exception for anything expected — none of that was ever exclusive to Rust, all of it was sitting in C# the whole time, and none of it felt urgent until Rust made thoroughness the default instead of the exception. It shows up in duller ways too: I profile before I guess now, I think about algorithmic complexity before I write the loop, I think about memory layout before I pick the collection. Not because C# suddenly requires any of that — because I finally do.

Miguel de Icaza — fellow Mexican, and someone who's been through more language migrations than most of us ever will — put it plainly back in 2019: *"All of us writing C and C++ are living on borrowed time. The only safe future is Rust. Prepare your code to go out of scope."*

That's the cycle closed. Not "I learned Rust." Something closer to: *I learned Rust, and it changed how I write the language I'm not leaving.*
