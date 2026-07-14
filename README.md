# loom_seed_mnist

Proof-of-concept: **train a dense MNIST classifier by searching `layer_seed` values only**.

Weights are never free parameters. Every weight matrix is always:

```text
layer_seed (uint64)  →  He-init PRNG  →  weight matrix
```

Save/reload is therefore a tiny JSON of seeds (~hundreds of bytes), not a weight dump.

```bash
cd loom_seed_mnist
./run.sh                 # menus for mode / reload / continue
./run.sh 5               # micro-fountain (per-digit micro → LT → mega)
./run.sh fresh 5         # wipe seeds/pop + micro-fountain
./run.sh 4               # dna-cascade
./run.sh continue        # RESUME search from mnist.seeds.json + mnist.pop.json
LOOM_SEED_MNIST_QUICK=1 ./run.sh continue
```

**Important:** a plain second `./run.sh` only **reloads** (no more searching).  
To pick up where DNA left off: `./run.sh continue` (or choose `[2] continue` in the menu).  
First menu is only reload/continue/fresh — train mode **[5]** is on the **next** screen (or pass `5` on the CLI).

---

## What this is (and isn’t)

| This PoC | Normal Loom training (`poly.Train`) |
|----------|-------------------------------------|
| Optimizes a few `uint64` seeds | Optimizes every weight |
| Weights always = `HeInit(seed)` | Weights drift freely via backprop |
| Checkpoint = seeds JSON | Checkpoint = weight blobs |
| Black-box search / evolution | Gradients |

It is **not** a replacement for SGD. It is a deliberate experiment: *how far can you push Loom’s deterministic seed→weight map as the entire “genome”?*

Related ideas elsewhere: evolutionary strategies, NEAT, hypernetworks, procedural generation. The Loom twist is the **hard constraint** that trained weights must remain expressible as seeded He-init.

---

## Data

1. Downloads classic MNIST IDX gzips into `data/` (Google CVDF / S3 mirrors).
2. Merges original 60k train + 10k test → **70 000** images.
3. Shuffles with a fixed Loom seed RNG.
4. Splits **80 / 20** → 56 000 train · 14 000 val.

Features: 28×28 pixels flattened → 784 floats in `[0,1]`.  
Target: digit `0…9` (argmax over 10 logits; one-hot used for MSE diagnostics).

Network (fixed for the PoC):

```text
784 → 128 → 64 → 10   (dense, float32, linear logits on last layer)
3 layer_seeds
```

---

## Five search modes

`./run.sh` asks which algorithm to use (after optional reload/continue/fresh).
Default interactive choice is **[5] micro-fountain**.

### [1] warmth — single genome · warm-bit hill-climb

One set of seeds. Each epoch:

1. Score a fitness batch with **soft fitness** (softmax NLL + margin) so the dial moves even when hard accuracy is stuck near chance.
2. For each layer, **probe all 64 bit flips** → Δsoft → a “warm bit” map.
3. Bias mutations toward warm bits; also try antithetic `+δ / −δ`.
4. Keep trials that improve soft fitness.
5. Periodically print Loom `DeviationMetrics` heatmaps on a val slice.

Intent: “hotter / colder” on the bit field of a single recipe. **One layer at a time** within each epoch.

### [2] dna — clustered multi-seed DNA attract (all layers · island model)

**K clusters** (default 4) each run a local population (default 6 → **24 genomes**).  
Each genome is `(s0,s1,s2)` — **all layer seeds move together**.  
Each cluster has its own elite `e*_k`. Clusters **migrate** explorers every few gens.

Uses Loom’s DNA engine (`poly.ExtractDNA` / `CompareNetworks`).

```text
per cluster k:
  collapse = mean pairwise DNA cosine inside cluster
  α = clip(0.10 + 0.55·gap + 0.20·(1-overlap), 0, 0.85)
  μ = clip(0.03 + 0.20·(1-overlap) + 0.25·collapse, 0.02, 0.35)
  s′ = s ⊕ ((s ⊕ e*_k) ∧ M_α) ⊕ M_μ
  if collapse > 0.85 → inject immigrants (fresh DNA)
  if collapse → 1 → shrink α (stop cloning elite)
hall-of-fame = best full val accuracy (not batch soft)
checkpoint → mnist.pop.json every generation
```

### [3] dna-layer — same DNA clusters, **one layer seed at a time**

Coordinate ascent: freeze `s_j` for `j≠ℓ`, run DNA clusters that only search `s_ℓ`, then advance to the next layer each round. Same attract/immigrant math; narrower gene edits.

### [4] dna-cascade — L0-heavy → expand free-set → warmth (new)

Hybrid that uses more of Loom’s DNA surface than [2]/[3]:

```text
Phase A  DNA islands search ONLY s₀ (heavy gens) — L0 was the only mover under [3]
Phase B  expand free-set {s₀,s₁} then {s₀,s₁,s₂}
         α/μ from CompareNetworks.LayerOverlaps per free layer (not OverallOverlap)
Phase C  warmth-bit polish on hall-of-fame; keep only if full val improves
```

Still on-manifold (`layer_seed → HeInit`). Aim: beat the ~18.3% dna-layer val record.

### [5] micro-fountain — per-digit micro → LT consolidate → mega

**No SGD.** Still seed-search + loom poly LT weight transport:

```text
for burst r = 1…R (default 3):
  μ) for digit c ∈ 0…9:
       short DNA islands scoring soft-fitness on class-c samples (+ small mix)
       keep into shared seed HOF only if full val rises (or micro-accept)
  seed-vote) majority bit-vote across the 10 regional genomes (on-manifold)
  κ) Pack(HeInit(s*_c)) → RecoverWeightBlobs → ensemble ForwardArgmax  (L1)
             stashes an L1 Master cargo
Μ) Level-2 RecoverWeightBlobs over padded L1 cargos → mega zoo ensemble

seeds.json = best seed-manifold genome (deployable HeInit)
fountain   = experimental weight-space ensemble (may leave seed manifold)
```

Does **not** call `poly.NeuralFountain` / `poly.Train`. Fountain codes only spray/peel already-packed HeInit blobs.

---

## Env knobs

| Env | Effect |
|-----|--------|
| `LOOM_SEED_MNIST_MODE=warmth\|dna\|dna-layer\|dna-cascade\|micro-fountain` | Select algorithm |
| `LOOM_SEED_MNIST_CONTINUE=1` | Resume from seeds + pop checkpoint |
| `LOOM_SEED_MNIST_QUICK=1` | Fewer epochs/gens, smaller batches |
| `LOOM_SEED_MNIST_EPOCHS` | Warmth epoch count |
| `LOOM_SEED_MNIST_MUT` | Warmth mutations per layer |
| `LOOM_SEED_MNIST_CLUSTERS` | DNA cluster count |
| `LOOM_SEED_MNIST_GEN` | DNA generations (affects micro gens too) |
| `LOOM_SEED_MNIST_BURSTS` | micro-fountain burst rounds |
| `LOOM_SEED_MNIST_ROUNDS` | dna-layer outer rounds |

CLI: `go run . --mode=micro-fountain` / `-5` / `--dna-cascade` / `--dna-layer` / `--dna` / `--warmth` / `--continue`.

---

## How to read the output

### Soft fitness

Continuous loss on logits. Used for within-batch ranking.  
**Lower = warmer.** Hard accuracy can jump around while soft slowly improves (or vice versa) because argmax is discrete.

### Thermometer / `soft↓%`

Soft-loss improvement vs the run’s starting soft. Often near empty on MNIST even when **hof val** climbs — soft batch noise ≠ full val. Trust **hof val** / full eval more.

### Warm bits (`b47(0.093)`)

Bits whose single flip *helped* on that epoch’s batch (warmth mode). `all cold` → no 1-bit flip beat current recipe that batch.

### DNA `cluster collapse` / `#1↔#2`

Per-cluster mean pairwise DNA cosine (and elite vs runner-up).  
- Near `0` → healthy diversity (what you want in mode [2]).  
- Near `1.0` with **all seeds free** → clones; immigrants should fire when collapse > ~0.85.  
- In mode **[3]**, collapse often sits ~0.6–0.7 even while searching: frozen layers share identical DNA, so cosine is inflated. Treat it as noisier here; trust hof / “UPDATED” vs “held”.

### `hof val` / `★ new hall-of-fame`

Best **full validation accuracy** seen so far. This is the real “are we better?” meter for DNA modes. Batch `acc=` lines are noisy lottery tickets.

### DeviationMetrics heatmap

Loom’s `poly.DeviationMetrics` / `PrintSummary` on a val (or train) slice:

- **Quality Score 0–100** — mean of `max(0, 100 - deviation%)`.
- **Buckets** — how wrong class predictions are.
- Exact digit hits → `0–10%` bucket + `█` on the strip.
- Random guessing ≈ 10% accuracy.

### Full eval lines

Accuracy on full train/val (56k / 14k). Trust these and hof val over fitness-batch `acc=`.

---

## What the runs are showing (PoC reading)

Illustrative numbers from real sessions — not SOTA claims.

### Warmth (epoch ~1→11)

| Checkpoint | Val-slice acc | Full val | Soft best |
|------------|---------------|----------|-----------|
| Before | ~10.2% | — | 2.55 |
| Epoch 5 | ~14.1% | ~14.9% | 2.38 |
| Epoch 10 | ~11.6% | ~12.3% | 2.37 |

**Reading:** Soft inches down; full val peaked ~15% then slipped. Warm bits went cold fast.

### [2] DNA clusters v2 (gen 0→26) — fresh run

Setup: `K=4 × 6 = 24` genomes, 30 gens, batch 384. Dense `784→128→64→10`.

| Checkpoint | Val-slice / heatmap | Full hof val | Soft↓ vs start | Diversity |
|------------|---------------------|--------------|----------------|-----------|
| Gen 0 | slice ~14.5%, Q~40.5 | **15.23%** | — | collapse 0.01–0.03, #1↔#2 ≈ 0 |
| Gen 4 | — | **16.79%** ★ | ~0% | still diverse |
| Gen 5 / 10 / 15 | slice ~16.4%, Q~40.4 | 16.79% (held) | 0–1% | cluster 2 sometimes #1↔#2≈0.67 |
| Gen 17 | — | **18.75%** ★ | ~1.8% | healthy |
| Gen 20 / 25 | slice ~18.9%, Q~42.8 | **18.75%** | ~1–2% | collapse mostly ≤0.03, `imm=0` |

**Reading:**

1. **Anti-collapse worked.** Unlike DNA v1 (cosine stuck at 1.0, frozen heatmaps), clusters stayed at collapse ≈ 0.01 with #1↔#2 near zero / slightly negative. Immigrants stayed at 0 because they weren’t needed — good.
2. **Hall-of-fame is the real scoreboard.** Soft thermometer barely moves (0–2%), but hof val climbed **15.2% → 16.8% → 18.8%**. Batch soft is a restless ranking signal; full val is the latch.
3. **Clusters explore different islands.** Early on cluster 3 held the soft elite; later cluster 2 often showed high batch acc (20–22%) while hof belonged to whichever genome won full val. That’s island-model behavior, not one global clone.
4. **Heatmaps finally track hof.** Quality ~40.5 → 42.8 and slice acc ~14.5% → 18.9% as hof advances. When hof holds steady (gens 5–15), the printed heatmap is identical — correct, not stuck search.
5. **Ceiling is still seed-manifold.** Peak ~19% val is the same ballpark as collapsed v1’s ~18%, but reached *without* killing diversity. So clustering fixed the failure mode (premature clone death) more than it unlocked a new accuracy regime — still three He-init seeds vs MNIST.

### DNA v1 (collapsed — contrast)

Old single-pop DNA hit ~18% val then `cosine = 1.0` forever; heatmaps froze while gens kept counting. v2’s job was to stop that. The [2] run above shows it did.

### [3] dna-layer (full 3 rounds) — coordinate DNA

Setup: `3 rounds × 3 layers`, `K=3 × 5` genomes per focus, **8 gens per layer**. Only one `layer_seed` free; the other two frozen.

| Step | Focus | Hof / full val | Notes |
|------|-------|----------------|-------|
| Start | — | **9.45%** | topology init (~chance) |
| Val BEFORE slice | — | slice 10.2%, Q 39.6 | |
| R1 · L0 | layer 0 | **15.23% → 17.38%** ★ UPDATED | first gene move |
| R1 · L1, L2 | mid / head | 17.38% held | batchAcc noise ≠ accept |
| After R1 | — | train 17.0% / val **17.38%** · slice 17.3%, Q 44.5 | |
| R2 · L0 | layer 0 | **17.38% → 18.31%** ★ UPDATED | second L0 win |
| R2 · L1, L2 | mid / head | 18.31% held | |
| After R2 | — | train **19.05%** / val **18.31%** · slice 18.2%, Q 41.2 | |
| R3 · all layers | — | 18.31% held every focus | plateau |
| Final | — | train **19.05%** / val **18.31%** | same as end of R2 |
| Init vs trained (slice) | — | val 10.3%→18.3% · train 10.6%→20.1% | clear over chance |
| Saved seeds | — | **only L0 `*`** · L1/L2 still topology init | ablation in the file |

**Reading:**

1. **Layer 0 did all the work — twice.** Hof path: **9.45% → 17.38% (R1 L0) → 18.31% (R2 L0)**. Round 3 found nothing better. Final JSON marks only layer 0 changed; L1/L2 remain original topology seeds. Ablation answer: on this dense net, the first-layer He-init recipe was the only gene that paid full-val rent.
2. **`batchAcc` ≠ accept.** Flashy mid-teens/20% batch acc is a **~384-row train lottery**. Updates need **full val** (`focusVal`) to beat hof. Soft/`★` at the *same* val can retune soft without raising hof → still `· layer held`.
3. **Round 3 = plateau, not a bug.** Same hof through another full L0→L1→L2 cycle. Seed-manifold ceiling here ~**18–19%** val — same ballpark as mode [2] (~18.8%), reached by editing **one** seed.
4. **`collapse≈0.67` under freezes** is shared DNA from frozen layers, not dna-v1 clone death. `imm=0` expected below the ~0.85 immigrant threshold.
5. **Heatmaps:** BEFORE Q 39.6 / 10%; after R1 Q 44.5 / 17%; after R2–3 Q 41.2 / 18% (quality can wobble while class acc rises). Comparison table: trained beats init on both splits.
6. **Vs [2]:** All-layers DNA ~18.8% with three movable seeds; dna-layer ~18.3% with effectively one seed moved. Similar ceiling, clearer *which*-seed story.

### [5] micro-fountain (3 bursts · full digits) — regional micro + LT + mega

Setup: `R=3` bursts, per digit `8` gens · free **L0 only**, regional fitness ~441 samples, LT loss 30%. Dense `784→128→64→10`. **No backprop.**

| Checkpoint | Metric | Notes |
|------------|--------|-------|
| Start | seed hof **9.45%** · slice ~10.2% | topology init |
| Burst 1 · digit 1 | seed hof → **16.73%** | ones specialist (regAcc ~76%); easy digit moves HOF first |
| Burst 1 · later digits | many `held` | sequential HOF gate: later digits rarely beat full val |
| Burst 1 · seed-vote | **16.73%** (held vs later HOF) | majority bit-vote of regionals can *hurt* |
| Burst 1 · κ L1 fountain | ensemble **19.89%** vs seed ~19.02% | first real fountain lift (~+0.9 pts) |
| Burst 2 · κ L1 | ensemble **18.69%** vs seed **19.39%** | averaging correlated cousins can go *down* |
| Burst 3 · digits 6–9 | soft↓ / batchAcc↑ on region, full held | e.g. digit 7 batchAcc ~70% but regAcc on val ~21%; digit 8 ~50% batch / 33% reg — skill exists, not kept in shared HOF |
| Burst 3 · κ L1 | ensemble **19.39%** = seed hof | flat |
| Μ mega (3 cargos) | ensemble ≈ **19.73%** (4k val) | best L1 fountain still **19.89%**; mega ≈ same regime |
| Final seed save | train **19.66%** / val **19.39%** | method `micro-fountain-v1` |
| Slice heatmap AFTER | val-slice **18.8%**, Q 45.3 | init→trained clear over chance |
| Genes that moved | **only L0 `*`** · L1/L2 still topology init | same ablation story as [3]/[4] |

**Reading:**

1. **Seed HOF beat the old ~18.3% record** (~**19.4%** full val). Mode [5] is a real seed searcher, not just fountain theater — and again **only `s0` changed**.
2. **Fountain helped once, modestly.** Best ensemble **19.89%** vs seed **19.02%** on burst 1. Burst 2 went *below* seed; mega landed ~**19.7%**. So LT consolidate-the-consolidations works as transport + soft blend, but is **not a second leap** off a ~20% seed ceiling.
3. **Sequential one-genome HOF starves specialists.** Digits like 7/8 show strong regional soft/batchAcc while `· held` on full val and low/medium `regAcc`. Regionals packed into L1 are often close cousins of the ones/global HOF → ensemble ≈ best member (or slightly worse). Majority seed-vote similarly failed to beat HOF.
4. **What would make fountain matter more:** keep a **true per-digit champion** (parallel region HOFs), pack *those*, then ensemble / gate / route — not overwrite one shared genome as you walk 0→9.
5. **Still no SGD.** Same ballpark as [2]/[3]/[4]. Fountain codes here recombine HeInit weight blobs; they do not train.

---

## Thoughts (honest)

1. **It “trains,” barely.** Chance (~10%) → ~19% with **three integers** is a real but tiny lift. Same MLP with backprop would crush 95%+.

2. **Seed space ≠ weight space.** He-init recipes are a tiny manifold. You’re searching recipes, not sculpting weights. Averaging packed recipes (fountain) only moves the needle when the recipes are *diverse specialists*.

3. **Warmth** = local compass on one genome; goes cold when 1-bit probes stop helping.

4. **DNA clusters [2]** = right *shape* for Loom’s DNA/NEAT story: diverse islands, hof on full val, heatmaps move with hof.

5. **dna-layer [3] is the ablation mode.** Full run: only `s0` ever UPDATED; `s1`/`s2` held across all rounds. Trust `hof val` / `UPDATED`, ignore flashy `batchAcc`. R3 plateau ≈ same accuracy regime as [2].

6. **micro-fountain [5]** pushes seed HOF slightly past [2]/[3] (~19.4%) and shows LT can recover regional packs + give a small ensemble bump once (~19.9%). Mega didn’t unlock a new regime. Bottleneck is **shared sequential HOF**, not LT peel failing (10/10 and 3/3 recovered).

7. **DeviationMetrics** is for humans; soft/hof for search. Don’t optimize the heatmap buckets directly.

8. **Portable recipe, not ImageNet.** Seeds-only + DNA compare + heatmaps (+ optional LT combo) is a coherent Loom demo stack. Competitive MNIST needs a richer genome (CNN seeds, more seeds, or actual `poly.Train` / NeuralFountain *specialize*).

---

## Layout

```text
loom_seed_mnist/
  main.go / run.sh
  seedmnist/
    download.go data.go train.go warmth.go
    dna_pop.go      # [2] clustered DNA + checkpoint
    dna_layer.go    # [3] coordinate DNA (one layer at a time)
    dna_cascade.go  # [4] L0-heavy → expand free-set → warmth
    dna_micro.go    # [5] micro seeds → L1 LT → mega
    run.go
  data/  mnist.seeds.json  mnist.pop.json
```

---

## Bottom line

> You can move MNIST accuracy with only Loom `layer_seed` → He-init weights, and show it with DeviationMetrics. Fountain codes can soft-combine those recipes, but only help a little unless the micro genomes stay diverse.

| Mode | Peak-ish full val (PoC) | What we learned |
|------|-------------------------|-----------------|
| warmth | ~15% | warm bits go cold |
| dna v1 | ~18% then freeze | population clones (`cosine=1`) |
| dna [2] clusters | **~18.8%** while diverse | hof > soft; islands work |
| dna-layer [3] | **9.5% → 17.4% → 18.3%** then plateau | **only L0 seed changed**; L1/L2 held |
| dna-cascade [4] | ~18.3% (record before [5]) | L0-heavy + LayerOverlaps; still L0-dominated |
| micro-fountain [5] | seed **~19.4%** · L1 ensemble **~19.9%** · mega **~19.7%** | seed beat ~18.3%; fountain +0.5–1 pt once; sequential HOF kills specialist diversity |
