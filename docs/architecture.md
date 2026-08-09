# MNIST NN Accelerator — Architecture

Target: 784 → 150 → 10 MLP classifier on KRIA KR260 (ZU3EG), running in PL with a
16×16 systolic array, a microcoded control unit (CU), resident weights in BRAM,
a dedicated POST (bias + ReLU + requant) stage, and an on-chip argmax that
outputs the digit index. PS handles top-level control and optional
softmax/confidence.

**Primary goal: fastest response for a single image** (inference, not training).

The Python/numpy side already exists: it trains/imports the model, quantizes to
int8, requantizes hidden activations with `shift = 11`, and emits `.mem` files
(weights, biases, instructions). This document specifies the PL-side
programming model so those files map 1:1 onto hardware.

## 1. Fixed-point / quantization model

- Input pixels: int8 (image already scaled to fit int8).
- Weights: int8, symmetric.
- MAC: int8 × int8 → int32 accumulate. **ACC_W must be 32**, the original 16-bit
  accumulator overflows for K=784.
- Hidden layer: acc32 + bias32 → ReLU → arithmetic right-shift by 11
  (`shift = 11`) → saturate/clamp to int8.
- Output layer: acc32 + bias32 → (linear, no ReLU) → output as-is. Prediction =
  `argmax(logits)`, computed in PL (§5). Softmax probabilities (for confidence)
  are optional and left to the PS — the model is trained with CrossEntropyLoss
  on raw logits, so the class is `argmax(logits)`; no softmax is needed for
  inference.

## 2. Latency model (why ~44 µs, and how to beat it)

Per-image latency = **MAC cycles, dominated by the K-stream**. For one image,
layer 1 needs 10 output tiles, each streaming K=784 + 31-cycle pipeline drain
→ ~8,150 cycles; layer 2 adds ~181. With per-tile POST/clear/fetch overhead:
**~8,750 cycles ≈ 44 µs @ 200 MHz** (88 µs @ 100 MHz).

- **Batch width ≠ latency.** The 16 lanes map to images; one image leaves 15
  lanes idle (or replicated), but the K-stream is unchanged → latency is
  identical for M=1 and M=16. Batching only scales throughput (16 images in the
  same 44 µs).
- **The real latency lever is PE count per image.** The streaming dataflow uses
  only 16 PEs per tile. A v2 weight-stationary-per-row design (each row holds a
  different tile's weights, stream `x[k]` once) reaches ~1,800 cycles ≈ **9 µs**
  — but requires PMAC weight registers, a load mode, and a loop-stack CU.
  Documented as v2 (§11), not v1.

**v1 decision: streaming dataflow, M=1 primary** (image replicated across all 16
lanes → uniform 128-bit banks and ISA), M=16 as an optional throughput mode with
identical latency.

## 3. Memory map

Separate single-port BRAM banks. A/B banks are 128-bit wide (16 × int8 per
word), mapped 1:1 to two 36Kb BRAM columns each.

| Bank             | Word width | ADDR_W | Depth  | Word content                                  |
|------------------|-----------|--------|--------|-----------------------------------------------|
| `instr`          | 64b        | 7      | ~128   | 1 instruction (64-bit)                        |
| `weights`        | 128b       | 13     | 7,990  | 16 weights (16 × int8), tile-major            |
| `bias`           | 32b        | 8      | 176    | 1 bias (int32) per neuron                     |
| `image`          | 128b       | 10     | 784    | word `k` = 16 lanes' pixel `k`                |
| `act`            | 128b       | 8      | 160    | word `j` = 16 lanes' activation @ neuron `j`  |

Word-layout conventions:

- **Weights (B), tile-major**: word `(tile, k)` = weights
  `[k][tile*16 + 0..15]`, i.e. 16 consecutive output neurons for a fixed input
  index `k`. Address = `tile*784 + k` so `MAC_TILE`'s `B_BASE + t` walk reads a
  tile's K words contiguously. Layer 1: 10 tiles × 784 = 7,840 words (raw data
  150→160 padded = 7,350 + 490 pad). Layer 2: 1 tile × 150 = 150 words.
  **Total 7,990 words ≈ 125 KB.**
- **Image / activations (A)**: 16-lane words. Word `k` = 16 lanes' value at
  index `k`. **M=1**: all 16 lanes carry the same pixel `x[k]` (replicated by
  the tooling/PS) → every array row computes the correct tile (redundant but
  correct). **M=16**: lane `i` = image `i` (transposed batch).
- **Act bank is transposed**: address = neuron index, word = 16 lanes → layer 2's
  A-stream reads addresses 0..149 contiguously. Layer 1 POST writes this layout;
  layer 2 MAC reads it. A single act bank suffices for serial v1 (logits
  overwrite `h` after layer 2 has consumed it); a second bank is reserved for
  inter-batch pipelining (§11).
- **Logits (output layer)**: POST with `QEN=0` packs 4 × int32 per 128-bit word;
  10 logits → 4 words (16 padded). Consumed by the on-chip argmax unit (§5); also
  readable by the PS if softmax/confidence is ever wanted.

Pads: layer 1 outputs 150 → 160 (10 tiles × 16); layer 2 outputs 10 → 16.
Layer 2 K = 150 (no pad needed). Zero-fill padded weights/activations in the
`.mem` generator so no special-case hardware is needed.

Byte order in 128-bit words: lane `i` at byte `i` (bits `8i+7:8i`).

Resources: weights ≈ 28 BRAM36K, image 3, act 1, bias 1, instr 1 → ~34 of
ZU3EG's 216 BRAMs. **256 DSP48E2** (one per PE; ZU3EG has 360).

## 4. Instruction set (microcoded CU)

Instructions are **64-bit** words; opcode in bits [2:0]. The CU is a
run-to-completion sequencer with a **linear (increment-only) PC** — the Python
tooling unrolls all tiles into a flat program. No loop stack in v1
(`SET_LOOP`/`END_LOOP` reserved for v2).

| Opcode | Name       | Fields                                                                 | Behavior |
|--------|------------|------------------------------------------------------------------------|----------|
| `000`  | `CLEAR_ACC`| —                                                                      | 1 cycle; pulse `acc_clr` + `skew_clr`. |
| `001`  | `MAC_TILE` | `A_SEL[3] A_BASE[14:5] B_BASE[27:15] K_CNT[39:28]`                     | Stream `K_CNT` A/B words into the array, accumulating. |
| `010`  | `POST`     | `QEN[3] DST_BASE[12:5] BIAS_BASE[20:13] SHIFT[25:21] RELU[26]`         | Per neuron: +bias, ReLU?, >>shift, clamp, write. |
| `011`  | `DONE`     | —                                                                      | Assert `done`, hold until next start. |
| `100`  | `NOP`      | `WAIT[10:5]`                                                           | Wait `WAIT` cycles. |

All operands are **immediates baked in by the tooling** — the CU never computes
addresses at runtime, only `base + t` walks. Field widths equal the bank depths
(A_BASE 10b → 1,024 ≥ 784/160; B_BASE 13b → 8,192 ≥ 7,990; K_CNT 12b → 4,096).

### `MAC_TILE`
Executes `t = 0..K+2N-2` (duration **K + 2N - 1**):
`A_addr = A_BASE + min(t,K-1)`, `B_addr = B_BASE + min(t,K-1)`.
`feed_en = 1` for `1 ≤ t ≤ K` (**K feed cycles**, BRAM 1-cycle latency folded in),
then a **2N-2 cycle drain** with `feed_sel = 0` (zeros keep the pipeline shifting
while the tail reaches the last PE; `feed_en` stays 1). The far PE (15,15)
settles at `t = K+30`. `A_SEL` selects the A-source bank: image (0) or act (1).
Run `CLEAR_ACC` first — accumulators keep their value across tiles.

### `POST`
Per neuron `j` (0..15), all 16 lanes: `s = acc[i][j] + bias[BIAS_BASE+j]`; if
`RELU`: `s = max(s, 0)`; if `QEN`: `s >>= SHIFT` (arithmetic), clamp to int8 →
pack 16 int8 → **1 word** at `DST_BASE + j`; else (logits) pack 4 × int32 →
**1 word per 4 neurons** at `DST_BASE + j/4`. ~2 cycles per neuron (bias prefetch
1 + write 1). No drain here — the array is already settled when MAC_TILE ends.

### Program (34 instructions)

```
# layer 1, tile = 0..9
CLEAR_ACC
MAC_TILE A_SEL=0 A_BASE=0 B_BASE=tile*784 K=784
POST     DST_BASE=tile*16 BIAS_BASE=tile*16 SHIFT=11 RELU=1 QEN=1
# layer 2
CLEAR_ACC
MAC_TILE A_SEL=1 A_BASE=0 B_BASE=7840 K=150
POST     DST_BASE=0 BIAS_BASE=160 SHIFT=0 RELU=0 QEN=0
DONE
```

## 5. Dataflow through the 16×16 systolic array

Array: `acc[i][j] += a_col[i] * b_row[j]` over `k`.
**i = lane/image (0..15), j = neuron-in-tile (0..15), k = input index.**

- A-word `k` = 16 lanes' pixel `k` (M=1: replicated; M=16: one image per lane).
- B-word `(tile,k)` = 16 weights `W[k][16t..16t+15]` (byte j = neuron offset).
- Each `MAC_TILE` computes a 16×16 tile = 16 lanes × 16 output neurons,
  accumulating over the full K (output-stationary, no partial-sum writeback).
- `CLEAR_ACC` only between output tiles (and fixes the original design, where
  `pmac` never cleared between runs).

Layer 1: 10 tiles × (K=784 + 31 drain) → ~8,150 MAC cycles. Layer 2: 1 tile ×
(150 + 31) → 181.

Cycle budget (per run — one image, or 16 at no extra cost):

| Phase | Cycles |
|-------|--------|
| Layer 1 MAC (10×815) | 8,150 |
| Layer 1 POST (10×33) | ~330 |
| CLEAR_ACC ×11 | 11 |
| Layer 2 MAC | 181 |
| Layer 2 POST | ~33 |
| fetch (~34 instr × 2) | ~68 |
| **Total** | **≈ 8,775** |

Latency: **≈ 44 µs @ 200 MHz** for one image (M=16 same latency, 16×
throughput). MAC-limited floor 7,990 cycles → ~91% array efficiency.

### Argmax (output stage)

The model is trained with CrossEntropyLoss on raw logits (`fc2(x)`), so the
predicted class is `argmax(logits)` — softmax is monotonic and only normalizes,
it never changes which logit is largest. The PL therefore classifies with a
**10-way argmax** over the layer-2 logits, no `exp` or division needed:

- **Sniffs the layer-2 POST writes** (QEN=0): argmax captures each 4×int32 logit
  word as it is written, reducing over the 4 words (16 slots, neurons 0–9 real).
- Comparator tree → 4-bit digit index, strobed together with `done`.
- Cost: ~10 cycles of overhead, zero BRAMs/DSPs (comparators only).

Softmax probabilities are **not** built into v1. If confidence values are ever
needed, the PS computes `softmax` from the same logits (cheap in C) — or a
LUT-based softmax can be added to the PL later (subtract max → exp LUT → sum →
reciprocal-multiply, ~50–100 cyc, 1 BRAM + a few DSPs, see §11).

## 6. Control: separate `pc` (fetcher) + `cu` (execute)

Two modules replace `sequencer.sv`. The **PC owns the program counter and the
instr-BRAM read**; the **CU decodes and schedules everything else**. This keeps
`top.sv` a thin wiring shell.

```
instr BRAM ◄─ pc_addr ── pc.sv ── ir/ir_valid ──► cu.sv ──► {bank addr/we,
                    (fetch pulse from cu)          │         feed_en/feed_sel,
                                                  │         acc_clr/skew_clr, done}
```

**`pc.sv` (fetcher):** on `start` → `pc=0`. On a `fetch` pulse from the CU →
place `pc` on the instr bank and advance; 2 cycles later pulse `ir_valid` with
the instruction on `ir_out` (BRAM 1-cycle read + 1 latch).

**`cu.sv` (executor):** run-to-completion FSM:

```
IDLE ─start→ FETCH1 → FETCH2 → RUN ─instr done→ FETCH1
                  │                      │
                  └─ opcode==DONE → DONE_ST
DONE_ST ─start→ IDLE
```

- `FETCH1`: assert `fetch` (1 cyc). `FETCH2`: latch `ir` on `ir_valid`; if
  `DONE` → DONE_ST, else → `RUN` with counter `t=0`.
- **No instruction pipelining**: macro-ops run hundreds of cycles and share the
  array/banks, so fetching the next instruction early gains nothing.
- Instruction timings: `CLEAR_ACC` = 1; `MAC_TILE` = K+2N-1 (feed 1..K + drain
  2N-2); `POST` ≈ 2 cyc/neuron (bias prefetch + write).
- Address generators walk `base + min(t,K-1)` during `MAC_TILE` (saturated),
  and `base + j` during `POST`.
- **POST datapath** is a 16-lane combinational `post` submodule — bias add →
  ReLU → arithmetic shift → clamp → pack 16 int8 (or 4×int32) → 128-bit write.
  **argmax** sniffs the QEN=0 writes (§5).
- v1 loads all banks via `bram` `$readmemh` INIT_FILE (preloaded at bitstream
  init) — no load port on `top`. Interface: `clk/rst/start/done` + 4-bit `digit`.

## 7. Required RTL changes from the current code

1. `pmac` / `systolic`: **N 4 → 16**, ACC_W 16 → 32; add `acc_clr` broadcast
   (also fixes the missing clear between runs). *(mostly done)*
2. `sequencer.sv` → replaced by **`pc.sv`** (fetcher) + **`cu.sv`** (decode/execute).
3. New **`post.sv`** (16-lane combinational): +bias → ReLU/linear → arithmetic
   `>>shift` → saturate → pack 16 int8 / 4×int32 → act bank.
4. New **`argmax.sv`**: sniffs layer-2 QEN=0 writes → 4-bit digit index.
5. `dual_bram.sv` → generic single-port `bram` with per-bank width/ADDR_W and
   `$readmemh` INIT_FILE; A/B banks 128-bit, bias 32-bit, instr 64-bit. *(done)*
6. `skew_buffer.sv` → N=16 and add `clr` (flush delay chains at tile
   boundaries, pulsed with `CLEAR_ACC`). *(done)*
7. `top.sv` → thin shell: 5 banks (instr/weights/bias/image/act) + pc + cu +
   N=16 array + post + argmax; interface `clk/rst/start/done` + 4-bit `digit`
   (no load port — banks preloaded from INIT_FILE).

## 8. AXI-Lite ↔ BRAM bridge (PS → PL) — short version

Add one AXI-Lite slave in `top` that gives the PS register access to fill the
banks and launch the CU. Minimum register set:

- `0x00 CTRL`   — bit0 `start`, bit1 `rst`, bit2 `load_en` (address decoder selects bank).
- `0x04 STAT`   — `done` (sticky, cleared on `start`), `busy`.
- `0x08 IADDR`  — target bank base address (instr/weights/bias/image).
- `0x0C IWDATA` — write-only data register; each write stores into the bank at
  `IADDR` then increments `IADDR`.

Flow: PS (a) writes `IADDR` = bank base + data words through `IWDATA` for all
banks, (b) writes `CTRL.start`, (c) polls `STAT.done`, (d) reads the 4-bit
argmax result (digit index) — and the raw logits only if softmax confidence is
desired.

Notes:

- No AXI-DMA needed — ~125 KB of weights is a few thousand AXI-Lite writes;
  fine for a one-time upload. Add AXI4/DMA later only if re-configuring weights
  per inference becomes a bottleneck.
- Give the AXI-Lite clock = `clk` (single clock domain) to avoid CDC entirely.
- Bank widths map 1:1 to BRAM columns (128-bit A/B = 2 × 36Kb columns).

## 9. Tooling ↔ hardware contract (what the `.mem` files must contain)

- `weights.mem`: tile-major, 16 int8/word (byte `j` = neuron offset within
  tile). Layer 1: 7,840 words; layer 2: 150 words. (Raw data = 7,350 words;
  layer-1 padding to 160 adds 490 words.)
- `bias.mem`: int32 per neuron. Layer 1: 0..159; layer 2: 160..169.
- `instr.mem`: the 34-instruction microprogram, 64-bit words.
- `image.mem`: 784 words of 16 lanes each. **M=1: replicate the pixel 16× per
  word.** M=16: transpose a 16-image batch (word `k` = lane `i` = image `i`'s
  pixel `k`).
- The TB references these `.mem` files for self-checking against reference
  outputs produced by the same Python script — including the final 4-bit
  `argmax(logits)` digit index.

## 10. Verification milestones

1. pc + CU + ISA reproduces the current 4×4 GEMM in sim (regression vs
   `tb_top.sv`).
2. N=16 array + POST + `acc_clr`; verify a single layer against numpy reference.
3. Full 2-layer MNIST with tooling-generated `.mem` files; accuracy self-check.
4. AXI-Lite bridge + argmax; run on KR260, PL outputs the digit index.

## 11. Future work

- **v2 latency cut (~9 µs)**: weight-stationary-per-row — PMAC weight
  registers, load mode, 784-cycle stream for all tiles; needs loop-stack CU.
- Inter-batch pipelining (layer 1 of next run overlaps layer 2 of current) — re-
  add a second act bank (ping-pong) and the `A_BANK`/`BANK_SEL` ISA fields.
- `SET_LOOP`/`END_LOOP` hardware loops (shrink program, batch looping).
- M=16 throughput mode (same latency, 16× throughput).
- LUT-based softmax in PL (exp LUT + divide) if confidence is needed without
  the PS (see §5).
