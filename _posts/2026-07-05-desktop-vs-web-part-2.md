---
title: "Desktop Apps Teach You More About a Language Than Web Apps Can — Part 2: Rust, FFI, and Concurrency"
description: "Cross the unmanaged border: Rust ownership, safety, FFI memory freeing traps, thread bounds, and resolving UI thread deadlocks inside Glance PDF."
Author: Jonas Lara
date: 2026-07-05 00:00:00 +0000
categories: [Software Engineering, Rust, Multithreading]
tags: [rust, dotnet, ffi, memory-safety, concurrency, winui, performance, glance]
image:
  path: /assets/img/post/desktop-vs-web-part-2/glance.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/desktop-vs-web-part-2/glance.png
  alt: Glance PDF logo
---

# Desktop Apps Teach You More About a Language Than Web Apps Can

> **Disclaimer:** This implementation was developed as part of a side project to explore and showcase Rust and C# integration and memory management in both. I am not a memory expert, so please keep that in mind.

In [Part 1](/posts/desktop-vs-web-part-1), we looked at how long-lived desktop states force you to master .NET memory structures: the stack, the heap, GC generations, and pinning through the lens of **[Glance PDF](https://github.com/jonas1ara/Glance)**.

But when your desktop application relies on a **hybrid architecture** (such as Glance's C#/.NET UI layer calling our high-performance native library written in Rust), you cross the border into the unmanaged world. 

Suddenly, your code has no safety net. A single mismatched pointer size, an unhandled thread race, or a wrong memory deallocation will crash the application instantly, leaving only a cryptic exit code in the OS logs.

This is Part 2, where we cover **Rust Ownership, Unsafe FFI boundaries, and UI Thread Deadlocks** in Glance.

---

## My First Rust App: The Journey to Memory Safety

**Glance PDF** is my first serious project written in Rust — shipped production code, not a toy exercise. 

As a .NET developer, the decision to step out of my comfort zone and learn Rust was driven by two main goals:
1. **Learn Rust by doing:** There is no better way to learn a system-level language than by building a real, high-performance product that interacts with native libraries.
2. **Explore [windows-rs](https://github.com/microsoft/windows-rs):** I wanted to test Microsoft's official Rust projection for Windows APIs. It is a toolset currently trending in Windows systems development as Microsoft actively works to rewrite components of the Windows kernel in Rust.

This decision was heavily influenced by a famous tweet from **Mark Russinovich** (CTO of Microsoft Azure), which declared:

> *"Speaking of languages, it’s time to deprecate starting new projects in C/C++ and use Rust for those scenarios where a non-GC language is required. For the sake of security and reliability, the industry should declare those languages as deprecated."*

To be clear, this is Russinovich's personal call to action about *new* projects, not an official Microsoft mandate—and the real Windows kernel work it echoes targets specific, security-sensitive components, not a wholesale rewrite of Windows in Rust. Still, the underlying argument is especially relevant when dealing with PDF files. Google's **PDFium** (which Glance uses for page rendering via Benoît Blanchon's [pdfium-binaries](https://github.com/bblanchon/pdfium-binaries) and the [pdfium-render](https://github.com/ajrcarey/pdfium-render) Rust crate) is a massive, highly optimized **C++ library**. While incredibly fast, C++ code is historically vulnerable to memory corruption, buffer overflows, and use-after-free bugs—which are the primary source of security exploits in document viewers.

### Rust as the Safe Bridge: Wrapping C++ for C#

Instead of exposing raw C++ PDFium pointers directly to our C# WinUI 3 frontend (which would leave our .NET runtime vulnerable to native heap crashes), we built a **Rust abstraction layer** (`glance-native`) around PDFium.

Rust acts as a **"Security Sandbox Guardian"**:

```
[ C# / WinUI 3 ] <=== Safe FFI === [ Rust Security Guard ] <=== Unsafe C++ === [ Google PDFium ]
```

1. **Compiler Sanitization:** The Rust wrapper wraps PDFium's raw C-pointers in type-safe Rust structures.
2. **Lifetime Enforcements:** Rust's compiler guarantees that unmanaged C++ pointers are never accessed after they are freed, preventing use-after-free bugs at compile-time.
3. **Data Race Prevention—A Deliberate Design Choice, Not a Compiler Guarantee:** Here's a catch worth being honest about. `PdfEngine` crosses the FFI boundary as an opaque raw pointer (`*mut PdfEngine`), and every `extern "C"` function dereferences it with `unsafe { &*engine }`. Once it's a raw pointer, it's no longer a type whose ownership Rust tracks, so its `Send`/`Sync` auto-traits have nothing left to check—the compiler simply can't see whether C# calls into it from two threads at once. So we made a deliberate choice instead of relying on the type system: enforce mutual exclusion at runtime, on the C# side, with a single `SemaphoreSlim` (`_ffiLock`) that serializes every call into the native engine. We'll get into exactly how that works in the concurrency section below.

By putting Rust in the middle, we create a secure, memory-safe translation layer. Rust sanitizes the raw C++ memory operations before exposing a clean, type-safe FFI boundary to C#, keeping Glance rock-solid and secure.

---

## 1. Rust Ownership: Safety Without a Garbage Collector

In C# or F#, memory is managed automatically by the Garbage Collector. In Rust, memory is governed by a strict set of compiler rules called the **Ownership Model**:

1. Each value in Rust has an owner.
2. There can only be one owner at a time.
3. When the owner goes out of scope, the value is dropped (deallocated) automatically.

### The Lifetime Challenge in Glance
In web development, a server handler is usually stateless. If you instantiate an object, it lives for the duration of a function call and gets dropped.

But in a desktop app like **Glance**, we instantiate a heavy native engine handle (`PdfEngineImpl` wrapping Google's PDFium library) and must **keep it alive** across multiple page views, user operations, and layout scroll cycles.

To share this state across the application boundary, Rust requires us to be very explicit about **lifetimes** (how long a reference remains valid) and **borrowing**:

```rust
// A wrapper representing the PDF engine state in Rust (glance-native)
pub struct PdfEngine {
    document: PdfDocument,
    current_page_index: usize,
}

impl PdfEngine {
    // We pass ownership of the document, which gets stored in the struct
    pub fn new(document: PdfDocument) -> Self {
        Self {
            document,
            current_page_index: 0,
        }
    }
}
```

If we pass a reference to the `PdfEngine` to a background thread pool worker to render a page, the Rust compiler will verify at compile-time that the engine isn't dropped by the main thread while the worker is using it. It forces you to write code that is structurally thread-safe before it ever compiles.

### When the Borrow Checker Isn't Enough: Self-Referential Structs
The simplified example above hides a real wrinkle. `PdfEngineImpl` (the actual struct in `pdf_engine.rs`) doesn't just hold a `PdfDocument`—it holds the `Pdfium` instance that the document was loaded from, in the *same* struct:

```rust
pub struct PdfEngineImpl {
    document: PdfDocument<'static>,
    _pdfium: Pdfium,
}
```

`PdfDocument` normally borrows from the `Pdfium` binding that created it, so its real lifetime is tied to `_pdfium`. But Rust's borrow checker can't express "this field borrows from that other field in the same struct"—that's the well-known self-referential struct problem, and it has no safe solution in stable Rust. Glance's escape hatch is an explicit unsafe cast:

```rust
// Safe transmute of PdfDocument lifetime to 'static since we hold the Pdfium instance
// inside the same struct and pdfium will be dropped after the document.
let document: PdfDocument<'static> = unsafe { std::mem::transmute(document) };
```

We didn't reach for this lightly. Crates like `ouroboros` or `self_cell` exist specifically to generate self-referential wrappers safely, and `Pin` solves a *related but different* problem—structs holding pointers into their own fields. Neither fits here: the borrow runs against the sibling `_pdfium` field, not against `self`, and pulling in a dependency to wrap one two-field struct felt heavier than a single, well-commented `unsafe` line we can audit directly. This is sound only because of a guarantee Rust does give us for free: struct fields drop in declaration order. Since `document` is declared before `_pdfium`, it's always dropped first—so the borrow the transmute papers over never actually outlives what it points to. The lesson isn't "the compiler protects you from every mistake automatically." It's that Rust makes you reach for `unsafe` deliberately, in one narrow, auditable spot, instead of silently accepting a use-after-free anywhere in the codebase the way C++ would.

---

## 2. The FFI Boundary: The Danger of `memory_free`

To make C# and Rust talk to each other, Glance uses a **Foreign Function Interface (FFI)** using C-compatible P/Invoke bindings. You can inspect all our signatures in [GlanceNative.cs](https://github.com/jonas1ara/Glance/blob/master/src/Interop/GlanceNative.cs).

```
[ C# / .NET Managed Heap ]  === P/Invoke ===>  [ Rust Unmanaged Native Heap ]
(Pointers: IntPtr)                             (Pointers: *mut u8)
```

At this boundary, the safety of both languages is disabled. You are dealing with raw, unmanaged pointers (`IntPtr` in C#, `*mut u8` in Rust). 

One of the most dangerous traps at this boundary is **memory deallocation**.

### The Heap Corruption Trap
Suppose Glance's Rust backend renders a PDF page into a PNG byte buffer and passes the raw pointer to C#:

```rust
// Rust: Allocates a PNG byte buffer on the native heap
#[unsafe(no_mangle)]
pub extern "C" fn render_page() -> Response {
    let png_bytes: Vec<u8> = render_pdf_to_png();
    let data_len = png_bytes.len() as u64;
    
    // Leaks the vector into a raw pointer so it persists after returning
    let raw_ptr = Box::into_raw(png_bytes.into_boxed_slice()) as *mut u8;
    
    Response {
        success: true,
        data_ptr: raw_ptr,
        data_len,
    }
}
```

Once C# finishes loading this PNG buffer into an image control, C# must free that unmanaged memory. 

If C# tries to free it using standard .NET allocators, or if Rust attempts to deallocate the pointer using a mismatched capacity, you corrupt the operating system heap. The next allocation will trigger an **access violation (`0xc0000005`)** and crash Glance instantly.

### The Solution: Reconstruct and Drop in the Creator's Context
The gold standard rule of FFI is: **The environment that allocates the memory must be the one that deallocates it**.

We implement a dedicated `memory_free` FFI endpoint in Rust. When C# is done with a buffer, it calls this endpoint, passing the pointer and the length. Rust reconstructs the original boxed slice with the exact same capacity and lets it drop naturally:

```rust
// Rust: Safely reconstructs the allocated type to release it
#[unsafe(no_mangle)]
pub extern "C" fn memory_free(ptr: *mut u8, len: u64) {
    if !ptr.is_null() {
        unsafe {
            if len == 0 {
                // Free as a null-terminated C-String
                let _ = std::ffi::CString::from_raw(ptr as *mut std::os::raw::c_char);
            } else {
                // Free as a boxed slice (capacity MUST match exactly)
                let slice = std::slice::from_raw_parts_mut(ptr, len as usize);
                let _ = Box::from_raw(slice);
            }
        }
    }
}
```

One subtlety worth flagging: `Box::from_raw(slice)` here isn't freeing a bare slice—`slice` is `&mut [u8]`, and Rust's reference-to-raw-pointer coercion turns it into the fat pointer `*mut [u8]` that `Box::from_raw` actually needs, carrying the exact length metadata `from_raw_parts_mut` just reconstructed. That's what makes the round-trip safe: the `Box<[u8]>` we drop here has the identical pointer+length layout as the one `Box::into_raw` produced back in `render_page`, even though the length crossed the FFI boundary as a separate `u64` argument instead of living inside the pointer itself. Get the length wrong and this reconstructs a differently-sized `Box<[u8]>`, corrupting the heap on drop—which is exactly why the length has to travel with the pointer everywhere it goes.

On the C# side, we import this function and invoke it as soon as we finish reading the bytes:

```csharp
// C# Import (GlanceNative.cs)
private const string DllName = "glance_native"; // .NET appends the platform extension (.dll) automatically

[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
public static extern void memory_free(IntPtr ptr, ulong len = 0);
```

---

## 3. Concurrency & UI Thread Dispatcher Deadlocks

In modern desktop frameworks (like WinUI 3), only the **main UI thread** is allowed to instantiate or update visual controls. 

If you try to render a page on a background thread pool worker, you must dispatch the resulting bitmap to the UI thread via a Dispatcher callback. This is standard.

However, mixing asynchronous ThreadPool tasks with native FFI locking leads to **circular deadlocks** if not designed with extreme care.

### The Deadlock Scenario in Glance
In Glance, a background task (`_backgroundRenderTask`) renders PDF pages sequentially using the Rust engine. To prevent concurrent access to the native engine, the background task and the UI thread share a semaphore lock (`_ffiLock` / `_ffiSemaphore`):

1. The **Background Thread** acquires the lock and calls `GlanceNative.pdf_render_page`.
2. The user decides to **Close** the application. The **UI Thread** enters the window close handler (`AppWindow_Closing`).
3. The UI thread cancels the background task and synchronously awaits it to finish.
4. But wait! The background task, while rendering, has queued a progress update to the UI thread via `DispatcherQueue.TryEnqueue`.
5. Since the UI thread is currently blocked awaiting the background task, it cannot process the dispatcher message queue.
6. The background task is blocked waiting for the dispatcher update to finish, and the UI thread is blocked waiting for the background task to finish.

**Result:** The application window freezes forever and Windows displays *"Glance (No Responding)"*.

```
[ UI Thread ] ──(Sync Await)──> [ Background Task ]
     ▲                                   │
     └───────(Dispatcher Callback)───────┘
          *DEADLOCK (Circular Block)*
```

### The Solution: Thread-Safe Disposal & Non-Blocking ThreadPool Offloading
To break this circular dependency, we did two things in Glance:

1. **Thread-Safe, Non-Blocking Engine Disposal:** 
   We modified [PdfRenderService.cs](https://github.com/jonas1ara/Glance/blob/master/src/Services/PdfRenderService.cs#L241) so `Dispose()` never blocks its caller. Instead of destroying the native Rust PDF engine handle synchronously, it wraps the teardown in `Task.Run(async () => { await _ffiLock.WaitAsync(); ... })`. If the background thread is mid-render, this fire-and-forget task waits on the ThreadPool—not the UI thread—until the page finishes, then calls `pdf_engine_destroy`, preventing access violations without ever risking a synchronous block on the caller.
2. **Non-Blocking ThreadPool Offloading on Exit:**
   We modified [MainWindow.xaml.cs](https://github.com/jonas1ara/Glance/blob/master/src/MainWindow.xaml.cs) to offload the shutdown and auto-save sequence to the **ThreadPool** entirely, letting the UI thread process its message queue freely. The real handler also decides *whether* to save and *how* to ask the user (a `ContentDialog` path when auto-save is off), but the deadlock-critical core is this:

```csharp
// MainWindow.xaml.cs: the deadlock-critical core of AppWindow_Closing
if (mainPage.HasUnsavedChanges)
{
    args.Cancel = true; // cancel synchronously to keep the UI message loop alive

    if (autoSave)
    {
        // Offload the save to the ThreadPool so the UI thread stays free
        // to pump its own dispatcher queue instead of blocking on it
        Task.Run(async () =>
        {
            try { await mainPage.SaveOriginalPdfSilentAsync(true); }
            catch (Exception ex) { Debug.WriteLine($"Failed to auto-save on close: {ex.Message}"); }

            _isClosingConfirmed = true;
            Environment.Exit(0);
        });
    }
    else
    {
        _ = ShowCloseDialog(mainPage); // prompts Save / Exit without saving / Cancel
    }
}
```

Whenever Glance actually needs to save on the way out, the save runs inside `Task.Run`, never synchronously on the UI thread, so the dispatcher queue stays free to process the very callback the background render task is waiting on. Once the background thread completes the save, it invokes `Environment.Exit(0)`, terminating the process cleanly and releasing all native file handles.

---

## Conclusion: The Desktop Art

Writing desktop applications is indeed a work of digital craftsmanship. It strips away the comforting abstractions of web frameworks and puts you face-to-face with CPU caching, memory alignment, OS message loops, and native heaps.

By bridging C# and Rust, Glance PDF gets the best of both worlds:
1. **C#/.NET:** A beautiful, responsive Fluent UI layer that integrates natively with Windows 11.
2. **Rust:** An ultra-fast, memory-safe backend engine that handles heavy lifting with zero runtime overhead.

The discipline required to make them work together safely and efficiently is what makes you a better, more complete programmer.

If you want to read the real-world implementation of these memory management patterns, check out the **[Glance GitHub Repository](https://github.com/jonas1ara/Glance)**.

***

### Support the Project!
* **Repository:** [github.com/jonas1ara/Glance](https://github.com/jonas1ara/Glance)
* **Get the App on Windows:** [Glance PDF on Microsoft Store](https://apps.microsoft.com/detail/9P387LMMCCTB)
