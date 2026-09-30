# Vektor — Comparison Demo Feature: Implementation Plan
## Live HNSW vs Brute-Force Comparison, Built on a Feature Branch

---

## Uncomfortable Truth First

[Certain] "The user enters data or documents" is the load-bearing phrase in your request, and it quietly requires an embedding model somewhere in the pipeline — Vektor has never accepted raw text, by explicit, repeated design decision going back to the original project scope ("Not an embedding generation service") and reaffirmed in the ecosystem context sync ("Do not build RAG. Build for RAG."). Adding text ingestion to `vektor.ui()` without drawing a hard boundary around where the embedding step lives will blur the one architectural line this project has held consistently since Phase 0.

The fix is not to avoid the feature — it's a genuinely strong demo idea — but to quarantine embedding generation entirely inside an optional, clearly-labeled demo layer that sits *on top of* Vektor Studio, never inside the core `vektor` package, and never as a silent dependency of `vektor[ui]`. Everything below is designed around that boundary.

Second point, and this is good news: you do not need "a brute-force database and an HNSW database." Brute-force search has no index — it is an exhaustive scan over stored vectors, and Vektor already stores every inserted vector's raw float32 data regardless of HNSW (that's `vector.bin` from Phase 5, independent of `graph.bin`). Phase 4's brute-force search already runs correctly against exactly the same stored vectors the HNSW graph is built from. There is no dual-indexing problem to solve — only a dual-*search* and comparison-*display* problem, which is considerably smaller in scope than what your description implies.

---

## Goal

Add a demo capability to Vektor Studio where a user submits documents, gets them embedded and inserted into a normal Vektor collection, then submits a query and sees brute-force and HNSW search results side by side with recall, latency, and speedup metrics computed live — built on a dedicated feature branch, touching the core `vektor` package only through its existing public API, and never blocking or altering the in-progress Phase 14 documentation and release work on `main`.

---

## Architectural Boundary — Decide This Before Any Code

Three layers, strictly separated:

**Core `vektor` package** — completely unchanged by this feature. It still only accepts vectors. No embedding logic, ever, lives here.

**`vektor[ui]` (Studio)** — also unchanged in its dependency footprint. Streamlit shell, existing pages, the `api_client.py` boundary — none of this needs to know embeddings exist.

**A new, separate optional layer — call it `vektor[demo]` or a wholly separate installable package like `vektor-studio-demo`** — this is where the embedding model lives, where "paste in documents" becomes "vectors," and where the comparison-specific UI page lives. [Likely] Packaging this as a fully separate optional extra (not bundled into `[ui]`) matters because the embedding model dependency (`sentence-transformers`, which pulls in PyTorch) is heavy — hundreds of megabytes — and installing it should be a clear, deliberate choice by anyone running the Studio for real operational use, not an accidental side effect of wanting the dashboard.

This mirrors the exact lazy-import discipline already built for `vektor.ui()` in the previous step: `vektor.ui()` works with zero embedding dependencies installed; the comparison demo page detects whether the demo extra is installed and shows a clear "install `vektor[demo]` to use this feature" message if not, rather than crashing the whole Studio.

---

## GitHub Branch Strategy

Create a feature branch off whatever branch currently holds the Studio work — name it something explicit like `feature/comparison-demo`, not off `main` directly if `main` is where Phase 14's documentation and release work is actively happening. [Certain] This feature must not block or interact with the PyPI release sequence from Phase 14 — that release is for the core package and REST API, which this feature does not touch at all. Keep the branches independent: Phase 14's release can ship v1.0.0 while this branch is still in progress, since the comparison demo is Studio-layer, optional-extra work with zero footprint on the core package being released.

Merge this branch back into the Studio's main development line only after its own checkpoint (below) passes — not on a deadline tied to Phase 14.

---

## Phase D0 — Boundary Setup and Branch Creation

**Goal:** Establish the branch and the three-layer dependency boundary before writing any feature logic.

**Tasks:** Create the feature branch. Add the new optional dependency group (`demo`, or the separate package, per the decision above) to packaging configuration, listing `sentence-transformers` as its dependency — [Likely] the specific model to use should be a small, CPU-friendly one such as `all-MiniLM-L6-v2`, which produces 384-dimensional embeddings, runs acceptably without a GPU, and is a completely standard, widely-used default for exactly this kind of demo — avoid reaching for a larger model that would make the demo slow or GPU-dependent for no benefit to what you're trying to show. Implement the same lazy-import-with-clear-error pattern from the previous package-boundary step, applied to this new dependency: attempting to use the demo features without the extra installed produces a named exception with an explicit install instruction, not a bare `ImportError`.

**Checkpoint:** `vektor[ui]` alone still works with zero embedding-related packages present; a clean environment with the `demo` extra additionally installed can successfully load the embedding model once.

---

## Phase D1 — Backend: Single-Query Comparison Endpoint

**Goal:** One new REST endpoint that, given a query vector already computed, runs brute-force search and HNSW search against the same collection and returns both results plus computed comparison metrics — this is pure reuse of existing Vektor internals, with zero embedding logic anywhere in this phase.

**Tasks:** Add `POST /v1/collections/{name}/search/compare` to the Phase 12 FastAPI layer. Internally, it calls the existing Phase 4 brute-force search and the existing Phase 8 HNSW search against the identical stored vectors, timing each with `time.perf_counter()` independently. It computes recall@k using the brute-force result as ground truth (exactly the same recall calculation already built in Phase 6, just applied to a single live query instead of a batch experiment) and returns a response containing: the brute-force result list, the HNSW result list, the HNSW's recall@k against brute force, the latency of each search path, and a speedup ratio (brute-force time divided by HNSW time).

[Certain] Do not let this new endpoint call the standard `/search` or `/search/compare` from within any code path that Phase 13's benchmark suite exercises — this comparison endpoint is demo/UX-only, separate and parallel to the production `/search` endpoint, so that adding it can never regress the latency numbers already measured and published in Phase 13's `Benchmark_Report.md`.

**Checkpoint:** Given a small test collection with a known set of vectors, the comparison endpoint returns correct brute-force and HNSW results, a recall@k value verifiable by hand, and both latency values populated and plausible (brute-force generally slower than HNSW at any meaningful collection size, though at very small sizes the two may be close or HNSW could even appear slower due to graph traversal overhead — document this as an expected, honest small-N edge case rather than treating it as a bug).

---

## Phase D2 — Backend: Document Ingestion Pipeline (Demo Layer Only)

**Goal:** Convert user-submitted text into vectors and insert them through Vektor's normal, unmodified public insert API — this is where the embedding model actually runs, entirely within the demo layer.

**Tasks:** Implement a function in the demo layer (not in `vektor` core, not in `vektor/ui/`) that accepts one or more text documents, runs them through the chosen embedding model, and calls `Collection.batch_insert()` — the exact same Phase 11 method every other ingestion path already uses — with the resulting vectors and the original text stored as metadata (so search results can display the actual document text, not just an ID and a score). [Certain] Storing the source text in the metadata dict is what makes the demo interpretable to a viewer — a bare vector ID means nothing to someone watching the comparison; seeing the actual matched sentence does.

**Checkpoint:** Submitting a handful of sample documents through this function results in them being retrievable via Vektor's ordinary `Collection.get()` and appearing in ordinary search results with their original text visible in metadata — proving this ingestion path is fully compatible with, and indistinguishable from, any other insertion path once vectors are in the collection.

---

## Phase D3 — Frontend: Document Ingestion Page (Demo Layer)

**Goal:** A Studio page — clearly labeled as part of the demo layer, not the core Dashboard/Collections flow — where a user pastes or uploads documents and watches them get embedded and inserted.

**Tasks:** Build a new page (a reasonable location is as a distinct mode within the existing "Import Data" page, or a wholly separate page such as "Comparison Lab" grouped under a clearly demo-labeled sidebar section — pick whichever fits your existing Studio information architecture better, but do not silently merge it into the plain vector-file import path, since that path has no embedding step and users should never be confused about which one they're using). Show ingestion progress reusing the existing structured event stream from the Studio's core build (Phase D2's batch insert already flows through the same insert path, so the same progress events from earlier Studio work apply here without new plumbing).

**Checkpoint:** A user can paste in, say, twenty short text snippets, watch them get embedded and inserted with visible progress, and then see them listed as ordinary vectors in the Collections page — confirming the demo ingestion path produces a perfectly normal Vektor collection underneath.

---

## Phase D4 — Frontend: Query and Compare Page

**Goal:** The centerpiece screen — a user types a query, and sees brute-force results and HNSW results side by side, with the comparison metrics from Phase D1 displayed clearly.

**Tasks:** Build the query input (text, run through the same embedding model from Phase D2 before being sent as a vector to the Phase D1 endpoint). Render two columns — brute-force results on one side, HNSW results on the other — each showing the matched document text, its score, and its rank. Above or beside the columns, display the computed metrics: recall@k, latency for each method, and the speedup ratio, ideally as a small, clearly-labeled stat row rather than a dense table, since the whole point of this screen is that the difference should be immediately legible at a glance.

[Likely] Consider visually highlighting results that appear in the HNSW list but not the brute-force ground-truth list (a miss) versus results that match — this turns an abstract recall number into something visually concrete, which directly serves your stated goal of demonstrating "the trade-off" rather than just stating a number.

**Checkpoint:** A live query against a demo collection of a few hundred to a few thousand documents produces a visibly readable comparison, with the recall number matching what you'd compute by hand from the two result lists, and the latency numbers reflecting a real, measured difference rather than a placeholder.

---

## Phase D5 — Scale Boundary and Honest Framing

**Goal:** Make sure the demo remains honest and responsive at the scale a live audience will actually use it at.

**Tasks:** [Certain] Brute-force search cost grows linearly with collection size — at a few thousand documents in a live demo it will feel instant either way, which risks the demo failing to show any meaningful latency difference at all. Decide deliberately what scale this demo targets: either cap the demo dataset at a size small enough that both approaches feel instant but the QPS/latency numbers still show a measurable, honestly-labeled difference in milliseconds even if a human can't feel it, or size the demo dataset large enough (tens of thousands of documents) that the difference becomes viscerally obvious, accepting that ingestion (embedding thousands of documents) will itself take real, visible time. [Guessing] A demo dataset in the range of a few thousand to low tens of thousands of documents is likely the sweet spot — large enough for a real, visible latency gap, small enough that embedding ingestion completes in a reasonable live-demo timeframe rather than requiring a long wait before the audience sees anything.

Add an explicit note in the UI itself — not just documentation — stating the dataset scale and being upfront that at very small scales the "brute-force is slower" story may not hold, consistent with intellectual honesty about what HNSW trades away at what scale, which is itself part of the research story this whole project has been telling since Phase 6.

**Checkpoint:** the demo's default dataset size and any presets offered produce a consistently honest, reproducible latency gap, verified by running the comparison multiple times and confirming the direction of the result doesn't flip due to noise.

---

## Phase D6 — Testing and Verification

**Goal:** Confirm the entire comparison pipeline is correct, not just visually plausible.

**Tasks:** Write a test that inserts a small set of known documents with hand-computable nearest neighbors for a given query, runs the comparison endpoint, and asserts the brute-force result matches the known correct answer exactly (since brute-force is exact by definition — this is the same ground-truth principle established all the way back in Phase 4) and that the reported recall@k is computed correctly against it. Write a test confirming the demo ingestion path (Phase D2) produces vectors indistinguishable from any other ingestion path — same dimension, same metadata structure, retrievable through every other existing Collection method.

**Checkpoint:** all tests pass; a person unfamiliar with the demo can run through Ingest → Query → Compare end to end following only the UI, with no manual setup steps outside what the Studio itself provides.

---

## Phase D7 — Branch Merge

**Goal:** Bring the completed feature back into the Studio's main development line.

**Tasks:** Confirm Phase 14's core package release (on its own, independent branch/timeline) has not been affected — re-run the core package's test suite and the Phase 13 benchmark suite unmodified, confirming zero regression, since this entire feature should have touched nothing in the core package's code paths. Merge the feature branch. Update Studio documentation to describe the comparison demo as an optional, clearly-labeled layer requiring its own extra install.

**Checkpoint:** the core package's CI remains green throughout, unaffected by this branch's existence; the merged Studio has the comparison demo available as an additional, clearly optional capability, not a default-on feature.

---

## What This Feature Deliberately Does Not Do

It does not make Vektor an embedding-generation service — the embedding model lives entirely in the optional demo layer, never in `vektor` core or in `vektor[ui]`'s mandatory dependency set.

It does not build a second, separate "brute-force database" — brute-force search runs against the same stored vectors HNSW already indexes; there is exactly one vector store, two search strategies over it.

It does not modify or add parameters to the production `/search` endpoint benchmarked in Phase 13 — the comparison endpoint is new and additive, isolated from anything Phase 13's numbers depend on.

---

## Assumptions

[Certain] This plan assumes your existing Studio information architecture (from the earlier Vektor Studio specification) has a sensible place to add one or two new pages without requiring a redesign of the whole navigation — worth a quick check before Phase D3 begins, since exactly where this lives in the sidebar affects how clearly it reads as "demo" rather than "core feature."

[Likely] `sentence-transformers` and its PyTorch dependency will be the single heaviest install in this entire project so far, likely exceeding the combined size of every other dependency across all 14 backend phases. This is expected and fine for an optional, clearly-labeled demo extra — but it reinforces why this must never become part of `vektor[ui]`'s baseline install.

[Guessing] The most persuasive version of this demo, given your stated goal of showing the trade-off rather than just the recall number, will come from Phase D5's honest scale-boundary work more than from any visual polish in D4 — a demo that shows a real, reproducible, honestly-labeled latency gap at a sensible dataset size will make your point more convincingly than a beautifully styled comparison view running against a dataset too small to show any real difference.
