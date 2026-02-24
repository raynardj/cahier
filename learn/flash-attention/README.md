Building Spec:

## Visualizations

- [Standard Attention](standard-attention.html) — Data movement between HBM and SRAM during standard self-attention
- [Flash Attention](flash-attention.html) — Flash Attention (Dao et al. 2022). S and P never touch HBM.

# Standard Attention

Build visualization for static page interactive effect.

It will be uploaded to the github page later on, so we should avoid libraries like React.

create a page standard-attention.html for understanding the standard-attention
* There should be a tiny box, um, upper center called SRAM
* There should be a huge box called HBM
* There are 4 lanes(memory controller) between SRAM and HBM

## Implementation requirements
* Matrices in HBM are made of square tiles (e.g. 128×16 matrix = 128 tiles wide × 16 tiles tall, 2048 tiles total)
* Tiles are copied block-by-block to SRAM for computation; results are copied back and accumulated in HBM
* Data transmission follows the memory controller lanes (vertical lanes between SRAM and HBM), not flying in all directions
* HBM area should be much larger than SRAM
* Everything is rendered on a single `<canvas>` element using the 2D drawing context (no SVG, no DOM-per-tile)
* Canvas is DPR-aware (retina-ready): logical size differs from backing-store pixel size
* Dark theme throughout (background #0f1117, surfaces #1a1d27, muted text, accent blue #6c8cff)

## Matrix and Vector Visualization

There are matrices and vectors could over lay HBM and SRAM, each float number is a tiny square with similar but darker color border, matrices and vectors are many of these squares stacked cleanly with very small spacings in between.

Usually matrices and vectors are smaller in dimension for the visualizetion, for hidden size, we use 16, for sequence length, we start with 128. We ignore batch size and number of heads for simpler visualization.

Each matrix is labeled with its ID and dimensions on the canvas (e.g. "X 128×16", "S 128×128"). For matrices S and P, each tile has a pre-generated random alpha value to give uneven brightness. P specifically has one bright "peak" per row to simulate a sharp attention distribution after softmax.

When tiles are being transferred out of HBM, they visually dim (alpha × 0.3) so the user can see which block is currently in transit.

## SRAM Layout

SRAM is divided into 3 labeled slots:
* Slot 0: **Operand A** — holds the left operand (e.g. an X block, a Q block, a P block, or S rows)
* Slot 1: **Operand B** — holds the right operand (e.g. a weight matrix Wq, a K block, or all of V)
* Slot 2: **Result** — holds the computation output before writing back to HBM

Matrix tiles inside SRAM slots auto-scale to fit the available slot area and display their label and dimensions.

## Computation.
There are computation state box near SRAM, container the current status text of the computation.

Computation only happens in SRAM. During computation the SRAM border flashes blue (a pulsing glow effect) to indicate work is happening on-chip.

### Data copying
tiny squares in HBM are in turn copied (new squares flying off them) to SRAM through the memory controllers, the squares should move in the lanes in some visible but still fast speed. (0.15 second per square for a lane).

After computation, square also moves the other way.

In many cases, the computation has to wait for data transfer to finish.

Transfer animation is direction-aware: squares move upward through lanes for toSRAM transfers and downward for toHBM transfers. The in-transit count is distributed evenly across all 4 lanes.

## Interactive Controls

The visualization is phase-based with 5 phases. The user controls playback via:
* **Play** — runs all phases from current position to end
* **Pause** — stops mid-phase (can resume with Play)
* **Step →** — advances exactly one phase
* **← Back** — rewinds to the previous phase (replays all prior phases instantly with animation skipped)
* **Reset** — returns to the initial state

Keyboard shortcuts: Space = play/pause, ArrowRight = step, ArrowLeft = back.

### Speed Modes

Two speed presets selectable via toggle buttons:
* **Fast (~20s)** — transfer 50ms, compute 40ms, pause 12ms (default)
* **Detailed** — transfer 600ms, compute 500ms, pause 150ms

### Cancellation

A generation counter (`runGen`) guards all async waits. Clicking Reset or Back increments the counter, causing any in-flight phase to throw a `'cancelled'` error and stop cleanly.

## Initial State in HBM

This 1st section is enclosed by a dashed box, says "Initial State in HBM".

* There is matrix X in the HBM in size of seq x hs, this is our input. color it white.
* Hidden layer for Query, Key and Value are in HBM too, the weights should be in size of hs x hs, and there are 3 of them, they are colored blue, green and red respectively.
There are 3 dashed squares encasing each matrix, above saying the phrase (weights for xxx).

## Projection of Query, Key and Value to hidden size

This 2nd section is enclosed by a dashed box, says "Projection of Query, Key and Value".

The following should be materialized in HBM after 1st round of computation.

* There is matrix Q in the HBM in size of seq x hs, this is our projected query. color it blue.
* There is matrix K in the HBM in size of seq x hs, this is our projected key. color it green.
* There is matrix V in the HBM in size of seq x hs, this is our projected value. color it red.

### Block-wise projection strategy
For each weight matrix (Wq, Wk, Wv in turn):
1. Load the full weight matrix (16×16) into SRAM slot 1 **once**
2. Loop over `NUM_BLOCKS` (= seq/BLOCK_ROWS = 128/16 = 8) blocks of X:
   - Load X block (BLOCK_ROWS×HS = 16×16) into SRAM slot 0
   - Compute the output block = X_block · W, result appears in slot 2
   - Write the result block back to HBM at the correct row offset
   - Clear slot 0
3. Clear slot 1 before moving to the next weight

## Computed pre-normalized attention scores

This 3rd section is enclosed by a dashed box, says "Computed pre-normalized attention scores".

The following should be materialized in HBM after 2nd round of computation.

* There is matrix S in the HBM in size of seq x seq, this is our computed pre-normalized attention scores. color it milky yellow (but in uneven brightness).

### Block-wise score computation
Double loop over Q blocks (i) and K blocks (j):
1. Load Q block i (BLOCK_ROWS×HS) into slot 0
2. For each K block j:
   - Load K block j (BLOCK_ROWS×HS) into slot 1
   - Compute S_tile = Q_block · Kᵀ_block / √d, result in slot 2
   - Write S tile (BLOCK_ROWS×BLOCK_ROWS = 16×16) back to HBM at position (i, j)
   - Clear slot 1
3. Clear slot 0 before next Q block

This produces the full seq×seq score matrix tile by tile.

## Computed attention scores

This 4th section is enclosed by a dashed box, says "Attention scores".

The following should be materialized in HBM after 3rd round of computation. Such softmax operation is done in SRAM, so we have to ship things to SRAM first, then ship it back to HBM.

* There is matrix P in the HBM in size of seq x seq, this is our computed attention scores. color it light indigo (but in uneven brightness, 1 of the color squares is much brighter than the others).

### Row-wise softmax strategy
Softmax must operate on full rows (the denominator sums across all columns). For each block of BLOCK_ROWS rows:
1. Load the full row strip of S (SEQ×BLOCK_ROWS = 128×16 tiles) into slot 0
2. Compute softmax across each row, result in slot 2
3. Write the P block back to HBM
4. Clear slot 0

## Computed attention output

This 5th section is enclosed by a dashed box, says "Attention output".

The attention output P and projected V are multiplied together(transport squares in and out of SRAM), output O in shape of seq x hs will be materialized in HBM.

### Block-wise output strategy
1. Load the **entire V matrix** (seq×hs = 128×16) into slot 1 **once** (it is the shared right operand for all blocks)
2. For each block of P rows:
   - Load P block (BLOCK_ROWS×SEQ = 16×128) into slot 0
   - Compute O_block = P_block · V, result in slot 2
   - Write O block (BLOCK_ROWS×HS = 16×16) to HBM
   - Clear slot 0
3. Clear slot 1

## Tally
* There should be total count of squares in HBM and SRAM (currently occupying), and the count of total squares transferred through the memory controllers.

Tally is drawn directly on the canvas (not as HTML elements):
* **SRAM count** — shown in cyan above the SRAM box, left-aligned
* **Transferred count** — shown in orange above the SRAM box, right-aligned
* **HBM count** — shown in green inside the HBM box, right-aligned

The tally is recomputed lazily (only when marked dirty by a transfer or slot change) and also during active transfers.

## Color Legend
A color legend strip is shown below the canvas mapping swatch colors to matrix identities:
X (white), Q/Wq (blue #6c8cff), K/Wk (green #4ade80), V/Wv (red #f87171), S (yellow #f7e4a3), P (indigo #9aa8ff), O (output #e2e4eb).

## Key insight note
S and P are the large N×N matrices that dominate HBM traffic in standard attention. Every square transferred is fully accounted for in the tally — this is what Flash Attention aims to eliminate.

# Flash Attention

Build visualization for `flash-attention.html` — same visual framework (single canvas, SRAM/HBM/lanes, dark theme, DPR-aware) but demonstrating the Flash Attention algorithm from Dao et al. 2022.

The visualization reuses the same dimensions (SEQ=128, HS=16, BLOCK_ROWS=16) and identical Phases 1–2 as standard attention so the viewer can directly compare. The critical difference begins at Phase 3: instead of materializing the full N×N matrices S and P in HBM, Flash Attention fuses score computation, softmax, and output accumulation in a single tiled pass entirely within SRAM.

## Same constraints as standard-attention.html
* Single `<canvas>` element, 2D context, DPR-aware
* Dark theme (#0f1117 background, same palette)
* Data moves through 4 memory-controller lanes, square by square
* Tiles are the atomic visual unit, every float transferred is tallied
* No external libraries (will be served on GitHub Pages)

## SRAM Layout — expanded from 3 to 6 labeled slots

Standard attention used 3 slots (Operand A, Operand B, Result). Flash Attention requires more simultaneous residents in SRAM. The SRAM box should be taller to accommodate:

| Slot | Label | Typical contents | Size |
|------|-------|-----------------|------|
| 0 | **K block** | K_j (one block of K, reused across inner loop) | B_r × d = 16×16 |
| 1 | **V block** | V_j (one block of V, reused across inner loop) | B_r × d = 16×16 |
| 2 | **Q block** | Q_i (one block of Q per inner iteration) | B_r × d = 16×16 |
| 3 | **O accumulator** | O_i (running output for this Q block, loaded & written back each iteration) | B_r × d = 16×16 |
| 4 | **Statistics** | m_i (row-max, B_r floats) and ℓ_i (row-sum, B_r floats) — drawn as two small column vectors side-by-side | B_r × 1 + B_r × 1 |
| 5 | **Scratch** | S_ij tile (B_r × B_c), then P̃_ij tile (same shape) — transient, never written to HBM | B_r × B_c = 16×16 |

This layout makes it visually obvious that S and P tiles are born and die inside SRAM (slot 5), never touching HBM.

## HBM Layout — no S, no P

The HBM region has **fewer** sections compared to standard attention — the N×N matrices that dominated the page are gone:

### Section 1 — "Initial State in HBM" (identical to standard attention)
* X (128×16 white), Wq (16×16 blue), Wk (16×16 green), Wv (16×16 red)
* Same dashed weight boxes with labels

### Section 2 — "Projection of Query, Key and Value" (identical to standard attention)
* Q (128×16 blue), K (128×16 green), V (128×16 red)
* Same block-wise projection strategy

### Section 3 — "Flash Attention Output"
This section replaces the three sections (S, P, O) from standard attention with a single section:
* **O** (128×16 output, colored #e2e4eb) — starts as all zeros (drawn as very dim/ghost tiles to show reserved space), gradually fills in with accumulated values
* **m** (128×1 column vector, colored cyan #22d3ee) — row-max statistics, initialized to -∞ (shown as empty/ghost)
* **ℓ** (128×1 column vector, colored orange #fb923c) — row-sum statistics, initialized to 0 (shown as empty/ghost)

There should be a visually prominent annotation near the missing S and P:
> A dashed outline labeled **"S (128×128) — NOT materialized"** and **"P (128×128) — NOT materialized"** drawn in ghost/very faint style below Section 3, to visually remind the viewer what Flash Attention avoids. These ghost outlines should be sized to scale so the viewer can appreciate the area savings.

## Phase 1 — Initial State (same as standard attention)
Matrices X, Wq, Wk, Wv appear in HBM. O, m, ℓ shown as ghost placeholders.

## Phase 2 — Projection of Q, K, V (same as standard attention)
Same block-wise strategy. Load each W matrix once, loop over X blocks, write Q/K/V blocks back. Identical transfer count.

## Phase 3 — Flash Attention Forward Pass

This is the core phase. It implements Algorithm 1 from the paper (FlashAttention-1 loop order: outer loop over K,V blocks, inner loop over Q blocks).

### Outer loop: for j = 1 to T_c (= 8 blocks of K and V)
1. **Load K_j** (16×16) from HBM → SRAM slot 0
2. **Load V_j** (16×16) from HBM → SRAM slot 1

### Inner loop: for i = 1 to T_r (= 8 blocks of Q)
3. **Load Q_i** (16×16) from HBM → SRAM slot 2
4. **Load O_i** (16×16) from HBM → SRAM slot 3  
   (on first visit, this is zeros; on subsequent visits, it carries the partial accumulation from previous j iterations)
5. **Load m_i, ℓ_i** (16×1 each) from HBM → SRAM slot 4  
   (on first visit: m = -∞, ℓ = 0)

6. **Compute S_ij = Q_i · K_j^T** in SRAM → result appears in slot 5 as the S tile (16×16, yellow #f7e4a3)  
   Status text: `S_ij = Q_i · K_j^T` — SRAM border flashes blue

7. **Compute online softmax statistics:**
   - m̃_ij = rowmax(S_ij)
   - P̃_ij = exp(S_ij − m̃_ij) — the S tile in slot 5 transforms into P̃ tile (indigo #9aa8ff)
   - ℓ̃_ij = rowsum(P̃_ij)
   - m_i^new = max(m_i, m̃_ij)
   - ℓ_i^new = e^(m_i − m_i^new) · ℓ_i + e^(m̃_ij − m_i^new) · ℓ̃_ij
   
   Status text: `Online softmax: update m, ℓ, rescale O` — SRAM border flashes blue  
   The statistics in slot 4 visually update (brief color pulse)

8. **Compute output accumulation:**
   O_i ← diag(ℓ_i^new)^−1 · (diag(ℓ_i) · e^(m_i − m_i^new) · O_i + e^(m̃_ij − m_i^new) · P̃_ij · V_j)
   
   Status text: `O_i += rescaled P̃_ij · V_j` — SRAM border flashes blue  
   The O accumulator in slot 3 pulses to show update

9. **Write O_i** (16×16) from SRAM slot 3 → HBM (overwrites previous O_i at correct row offset)
10. **Write m_i, ℓ_i** from SRAM slot 4 → HBM (update statistics)
11. **Clear slots 2, 3, 4, 5** (Q_i, O_i, stats, scratch done for this i)

### End inner loop
12. **Clear slots 0, 1** (K_j, V_j done for this j)

### End outer loop

After the double loop completes, O in HBM contains the final attention output. m and ℓ are no longer needed (they were auxiliary).

### What happens in slot 5 (Scratch)
The S tile and P̃ tile are computed and consumed *entirely within SRAM*. They never generate any HBM transfer. This should be visually emphasized:
- When the S tile appears in slot 5, show it briefly in yellow
- When it transforms to P̃, crossfade to indigo
- When P̃ is multiplied with V_j to update O_i, the tile fades away
- At no point do squares fly down the memory-controller lanes from slot 5

## Tally

Same tally display as standard attention (SRAM count, HBM count, Transferred count on canvas).

### Expected transfer comparison
With SEQ=128, HS=16, BLOCK_ROWS=16, T_r=T_c=8:

**Standard attention total transfers** (from the existing visualization):
- Phase 2 (projection): same
- Phase 3 (S = QK^T): reads of Q and K blocks + writes of S tiles = many N×N tile transfers
- Phase 4 (softmax): reads of full S rows + writes of P rows = 2 × N×N
- Phase 5 (O = PV): reads of P blocks + V + writes of O = N×N + N×d + N×d

**Flash Attention total transfers** (Phase 3 only replaces Phases 3–5 of standard):
- Outer loop reads of K_j, V_j: T_c × 2 × (B_c × d) = 8 × 2 × 256 = 4,096
- Inner loop reads of Q_i, O_i, m_i, ℓ_i: T_c × T_r × (B_r×d + B_r×d + B_r + B_r) = 8 × 8 × (256+256+16+16) = 34,816
- Inner loop writes of O_i, m_i, ℓ_i: T_c × T_r × (B_r×d + B_r + B_r) = 8 × 8 × (256+16+16) = 18,432
- **No S or P transfers at all** — they live and die in SRAM

The exact numbers should be computed programmatically and displayed at the end. The key message: the total transferred count should be noticeably smaller than standard attention, despite doing the same mathematical work.

## Comparison Summary Panel

After Phase 3 completes, display a summary comparison below the canvas (drawn on canvas or as HTML):

| Metric | Standard Attention | Flash Attention |
|--------|-------------------|-----------------|
| Peak HBM usage | includes S (128×128) + P (128×128) = 32,768 floats | **No S or P in HBM** |
| Total floats transferred | (number from standard) | (number from flash, lower) |
| Extra FLOPs | — | Recomputes S, P̃ per tile (traded for less I/O) |

## Backward Pass Note

After the visualization completes, show a brief text note (on canvas or as HTML below):

> **Backward pass (not shown):** In standard attention, S and P are stored in HBM during the forward pass so the backward pass can read them. Flash Attention does **not** store S and P — instead, it **recomputes** them from Q, K, V blocks during the backward pass. This trades O(N²) extra FLOPs for O(N²) less HBM storage and I/O. Since attention is memory-bound (not compute-bound), this trade is net faster.

This is mentioned for completeness but this page focuses on the forward pass.

## Interactive Controls (same as standard attention)
* Play, Pause, Step →, ← Back, Reset
* Fast / Detailed speed presets
* Keyboard shortcuts: Space, ArrowRight, ArrowLeft
* Same cancellation mechanism (runGen counter)

## Color Legend
Same as standard attention, but with additions:
* X (white), Q/Wq (blue), K/Wk (green), V/Wv (red), S tile (yellow, SRAM only), P̃ tile (indigo, SRAM only), O (output #e2e4eb), m (cyan), ℓ (orange)
* S and P legend entries should have a note: "(SRAM only — never in HBM)"

## Key Insight Notes

1. **"S and P never touch HBM."** — The N×N score and probability matrices exist only as transient B_r × B_c tiles inside SRAM. This eliminates the O(N²) HBM storage and the O(N²) read/write traffic that dominates standard attention.

2. **"Same math, less data movement."** — Flash Attention computes the exact same result as standard attention. The only difference is *where* intermediates live. By keeping S and P on-chip, the total bytes moved through the memory controller drops significantly, and since attention is memory-bandwidth-bound, wall-clock time drops proportionally.

3. **"Online softmax makes it possible."** — Standard softmax needs the entire row of S to compute the denominator. The online softmax trick (Milakov & Gimelshein 2018) maintains running statistics m (row-max) and ℓ (row-sum-of-exponentials) that are incrementally updated as each K block is processed. This is what allows processing S one tile at a time instead of one full row at a time.