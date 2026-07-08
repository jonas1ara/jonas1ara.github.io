---
title: "Desktop Apps Teach You More About a Language Than Web Apps Can — Part 1: .NET and the GC Dance"
description: "In web development, memory is a request-scoped abstraction. On the desktop, state lives forever. Let's dive into .NET memory internals (Stack, Heap, LOH, and Pinning) through the lens of Glance PDF."
Author: Jonas Lara
date: 2026-07-04 00:00:00 +0000
categories: [Software Engineering, .NET, Desktop]
tags: [dotnet, memory-management, garbage-collector, csharp, performance, glance]
image:
  path: /assets/img/post/desktop-vs-web-part-1/glance.png
  lqip: https://raw.githubusercontent.com/jonas1ara/jonas1ara.github.io/refs/heads/main/assets/img/post/desktop-vs-web-part-1/glance.png
  alt: Glance PDF logo
---

# Desktop Apps Teach You More About a Language Than Web Apps Can

> **Disclaimer:** This implementation was developed as part of a side project to explore and showcase Rust and C# integration and memory management in both. I am not a memory expert, so please keep that in mind.

In modern web development, memory is largely a request-scoped abstraction. A request comes in, a handful of short-lived objects are allocated, a database query is resolved, a JSON response is serialized, and the Garbage Collector (GC) silently sweeps away the debris. Since web servers are stateless and request loops are brief, you can get away with poor memory practices for a very long time before your server runs out of RAM.

But on the desktop, **state lives forever**.

A desktop application (like a document viewer or editor) is a single, long-lived process. If you allocate a 10MB bitmap to display a page, and fail to release its reference or manage its layout virtualization, that memory is leaked. If your scroll viewer triggers GC collection sweeps every time the user scrolls, the main UI thread will stutter, dropping frames and destroying the user experience.

Writing a high-performance desktop application forces you to look under the hood of your language runtime. It turns abstract concepts into hard execution constraints.

This is Part 1 of a two-part series on what building **[Glance PDF](https://github.com/jonas1ara/Glance)**—a hybrid desktop PDF reader ([WinUI 3](https://github.com/microsoft/microsoft-ui-xaml) + Rust)—teaches you about systems-level programming. Today, we're diving deep into the **.NET Memory Model: Stack, Heap, and the GC Dance**.

---

## The Stack vs. The Heap: The First Line of Defense

In .NET, memory is divided into the Stack and the Managed Heap. In typical web development, we write code without ever thinking about where our variables land. But when building high-frequency UI pipelines (such as rendering PDF page grids or indexing search coordinates), heap allocations become your worst enemy.

### The Stack: High-Speed, Zero-Cost Memory
The Stack is a fast, sequential scratchpad. Allocations on the stack are practically free—just a single CPU instruction shifting the stack pointer. When a method exits, its stack memory is reclaimed instantly. No GC, no cleanup overhead.

In .NET, we can exploit the stack using value types (`struct`), `ref struct`, and `Span<T>`.

### Case Study: Crossing the FFI Boundary Without Allocating a Wrapper Object
Every time Glance asks the Rust backend to render a page, it has to cross the P/Invoke boundary hundreds of times per scrolling session. If each call allocated a heap object just to carry a handful of `uint`s and a `float` into Rust, that's a Gen 0 allocation for every single page render—thousands over the lifetime of a session.

Instead, the render request is a `struct`, not a `class`:

```csharp
// GlanceNative.cs
[StructLayout(LayoutKind.Sequential)]
public struct RenderOptions
{
    public uint PageIndex;
    public uint Width;
    public uint Height;
    public float Dpi;
}

[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
public static extern Response pdf_render_page(
    IntPtr engine,
    ref RenderOptions opts,
    out IntPtr outPngData,
    out ulong outPngLen
);
```

Because `RenderOptions` is a value type, `ref RenderOptions opts` passes a pointer to stack memory straight into the native call—no boxing, no heap object for the marshaller to allocate and the GC to later collect. The matching `Response` struct that comes back (`Success`, `ErrorMsg`, `DataPtr`, `DataLen`) follows the same rule: it's a `[StructLayout(LayoutKind.Sequential)]` struct that mirrors the Rust `#[repr(C)]` layout byte-for-byte, so the marshaller can copy it directly without reflection.

```csharp
// PdfRenderService.cs
var response = GlanceNative.pdf_render_page(_pdfEngineHandle, ref opts, out pngDataPtr, out pngDataLen);
```

The PNG bytes themselves still have to land on the managed heap eventually (`byte[] pngBytes = new byte[pngDataLen]`, followed by `Marshal.Copy`)—there's no getting around that when you need a `byte[]` to hand to WinUI's image pipeline. But the *request/response envelope* that travels the hot path on every render call stays off the heap entirely. That's the actual lesson: not every FFI payload needs to be zero-allocation, but the ones you cross the boundary with on every frame absolutely should be.

---

## The GC Dance: Generations and the Large Object Heap (LOH)

When we do allocate on the heap, .NET uses a generational Garbage Collector to reclaim memory. The GC divides the heap into three generations:

```
[ Gen 0 ] ---> [ Gen 1 ] ---> [ Gen 2 ] ---> [ Large Object Heap (LOH) ]
(Short-lived)  (Buffer Zone)  (Long-lived)   (Allocations > 85,000 bytes)
```

* **Generation 0:** Holds short-lived objects (temporary variables). Collects are frequent and extremely fast.
* **Generation 1:** Serves as a buffer zone between short-lived and long-lived objects.
* **Generation 2:** Contains long-lived objects (global state, view models, application caches). Gen 2 collections are expensive and can freeze the application (Stop-the-World pause).

### The Large Object Heap (LOH) Trap in Glance
There is a fourth region: the **Large Object Heap (LOH)**. Any object larger than **85,000 bytes** bypasses Gen 0 and Gen 1 entirely and is allocated directly on the LOH. 

Why does this matter for a desktop app like **Glance**?
In Glance, the sidebar displays high-definition page thumbnails dynamically. As the user scrolls, the application renders page bitmaps on the fly. Each raw thumbnail bitmap is a byte array easily exceeding 200KB. 

If we allocated a new `byte[]` array for every page thumbnail, they would all land on the LOH. 
1. **No Compaction by Default:** The GC does not compact the LOH on every pass, because copying huge blocks of memory is CPU-intensive. This leads to LOH fragmentation, bloating Glance's memory footprint. Since .NET 4.5.1 (and readily available in every version Glance targets), you can opt in to a one-time compaction with `GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce` before forcing a `GC.Collect()`—useful as a last resort after a bulk operation, but not something you want to lean on in a scrolling hot path.
2. **Tied to Gen 2, Not Gen 0/1:** The LOH is swept only as part of a full Generation 2 collection—there's no separate "Gen 0 for large objects." It's not that LOH allocations directly *force* a Gen 2 collection to happen; it's that every object, large or small, is collected on whatever generation's schedule it belongs to, and the LOH's schedule is Gen 2's. So frequent large allocations inflate the LOH between sweeps, and when a Gen 2 collection *does* trigger (from Gen 0/1 promotions filling their budget, or an explicit `GC.Collect()`), it now also has a bloated, fragmented LOH to walk through—making an already-expensive pause worse.

In a web API, a Gen 2 sweep adds 50ms to a request—unfortunate, but rarely noticeable. In a desktop app, a 50ms pause during scrolling drops the frame rate from 60fps to 15fps. The UI visibly stutters.

### The Solution: Glance's Dual-Engine Architecture

To prevent LOH fragmentation and keep memory usage flat, Glance implements a **hybrid dual-engine architecture**:

1. **The Layout & Metadata Engine (Managed .NET / Windows API):**
   * Upon opening a document, Glance loads the structure using native Windows UWP APIs (`Windows.Data.Pdf.PdfDocument`) via an in-memory stream.
   * This managed Windows API is extremely fast at parsing page counts, dimensions, and metadata. It runs on the UI thread, instantly setting up the virtualized scroll layouts and placeholders so the user sees the document container structure immediately.
2. **The Rendering & Search Engine (Unmanaged Rust + PDFium):**
   * For resource-heavy tasks—like high-definition page rendering, text selection highlight coordinate mapping, and full-document text searches—Glance offloads the work to our custom Rust library (`glance_native.dll`) which wraps Google's PDFium.
   * C# delegates these tasks asynchronously to unmanaged thread pools (`AsyncBridge.RunNativeAsync`), returning raw byte buffers from the native heap only when a page enters the viewport.

By splitting the workload, C# handles the visual container recycling on the UI thread, while Rust handles the heavy bitmap allocations entirely on the unmanaged native heap. This keeps the managed .NET GC heap clean, preventing Large Object Heap fragmentation and ensuring a buttery-smooth 60fps scrolling experience.

---

## Pinning: The Problem Glance Sidesteps Entirely

When you write a desktop application that integrates with a native backend (like a C++ or Rust library), you frequently pass data back and forth.

However, the .NET Garbage Collector is **relocating**. During a collection sweep, the GC moves objects around the heap to compact memory and update references.

If you pass a raw pointer to a managed C# `byte[]` array to a Rust library, and the GC moves that array while Rust is reading from it, the native pointer becomes invalid. Rust will read garbage memory or write into a random address, causing a **segmentation fault** or **access violation exception (`0xc0000005`)** that instantly crashes the app.

The textbook fix is to **pin** the memory with an `unsafe fixed` block so the GC promises not to move it for the duration of the call. Glance never actually needs to do this, and the reason why is worth understanding.

For the annotation persistence path, Glance passes a JSON `string` straight across the P/Invoke boundary:

```csharp
// GlanceNative.cs
[DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
public static extern Response annotations_save(string json, string path);

// PersistenceService.cs
var response = GlanceNative.annotations_save(annotationJson, filePath);
```

A managed `string` isn't blittable—it's UTF-16 internally, and Rust expects a null-terminated byte buffer—so the CLR's built-in marshaller converts it for you. Per the declared `CharSet` (here `CharSet.Ansi`), it marshals the `string` to an `LPSTR` (or `LPCWSTR` for `CharSet.Unicode`): it allocates a temporary unmanaged buffer, copies the string into it in the target encoding, and hands Rust a pointer into memory the GC was never tracking in the first place. Technically the marshaller *does* pin the original managed string for the instant it takes to read its characters during that copy—but that pin is scoped to the copy itself, not the whole native call, and it's handled entirely inside the marshaler with no `fixed` block for us to write. There's no `byte[]` to pin for the duration of the call because there's no managed array on the other side at all—just a marshaller-owned unmanaged buffer that gets freed once `annotations_save` returns.

The same logic applies to the render and search calls from earlier: `RenderOptions` is a blittable `struct`, so it's copied by value onto the native stack frame, and the PNG/JSON bytes that come back travel the other direction—from Rust's native heap into a fresh managed `byte[]` via `Marshal.Copy`, after the fact, not while Rust still holds a reference to it.

**Glance's project file doesn't even have `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>` set**—the codebase is 100% safe C#. That's not a limitation; it's the payoff of designing the FFI surface (strings, blittable structs, and copy-after-the-fact byte arrays) so that manual pinning is never required. Reaching for `fixed` is the right call when you must share a live managed buffer with native code for the duration of a call—but the better first question is whether you can restructure the boundary so nothing needs to be shared live at all.

---

## Conclusion & Code Reference

Understanding the stack, the heap, generations, and the mechanics of GC pinning is what transforms a junior developer into a systems-aware software engineer. It forces you to write code that respects the hardware.

If you want to read the real-world implementation of these memory management patterns, check out the **[Glance GitHub Repository](https://github.com/jonas1ara/Glance)**:
* Review [PdfRenderService.cs](https://github.com/jonas1ara/Glance/blob/master/src/Services/PdfRenderService.cs) to see how native Rust FFI calls are bound and managed.
* Examine [AsyncBridge.cs](https://github.com/jonas1ara/Glance/blob/master/src/Interop/AsyncBridge.cs) to see how we bridge .NET tasks with native thread pools.

In **Part 2**, we will cross the boundary into the unmanaged world:
* How **Rust's Ownership and Lifetime models** manage memory without a Garbage Collector.
* How C# and Rust communicate safely via **Unsafe FFI boundaries**.
* The mechanics of **thread safety** and resolving **UI thread dispatcher deadlocks**.

Stay tuned for [Part 2](/posts/desktop-vs-web-part-2)!

***

### Support the Project!
* **Repository:** [github.com/jonas1ara/Glance](https://github.com/jonas1ara/Glance)
* **Get the App on Windows:** [Glance PDF on Microsoft Store](https://apps.microsoft.com/detail/9P387LMMCCTB)
