package seedmnist

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/openfluke/loom/poly"
)

/*
Sample-fountain (mode 6) — sample-shard micro → LT → mega:

  Mode 5's digit partitioning starved diversity (ones locked a shared HOF).
  Mode 6 partitions *random training samples* into shards and keeps
  independent specialist genomes for the fountain pack — full-val HOF is
  only for deployable seeds.json.

  for burst r = 1…R:
    μ) reshuffle train → K shards of N samples
       each shard: short DNA islands on soft-fitness(mix(shard, global_anchor))
       ALWAYS keep s*_i for packing (even if full val does not rise)
       update seed HOF when a specialist does beat full val
    κ) Pack all specialists → RecoverWeightBlobs →
         ensemble avg + soft-gated top-k ensemble
    polish) warmth on seed HOF (keep if full val rises)
    stash L1 cargo

  Μ) mega over L1 cargos

Ideas aimed past ~19.7%:
  • sample shards ≠ digit labels → more complementary specialists
  • independent pack members (no sequential overwrite for fountain)
  • mixed fitness (shard + global anchor) so specialists stay slightly global
  • free-set expands across bursts: {0} → {0,1} → {0,1,2}
  • soft-gated top-k ensemble (drop weak cousins before averaging)
  • warmth between bursts on seed HOF
  • pairwise DNA pull among top shard elites (seed-side), not crude majority of all
*/

func printSampleFountainEquation() {
	fmt.Println(`
── Sample-fountain (mode 6) — sample shards → LT → mega ──
  Why not digits: mode 5 locked a ones-biased shared HOF; later digits held.
  μ) random train shards (reshuffled each burst) · independent s*_i for pack
     fitness = soft(mix(shard, global_anchor))  · free-set expands over bursts
  κ) Pack(HeInit(s*_i)) → RecoverWeightBlobs → avg + soft-gated top-k
  polish) warmth on seed HOF between bursts
  Μ) consolidate L1 cargos (mega)
  Beat ~19.7%: diversity in the pack + gated blend + expand free layers`)
}

func trainSampleFountain(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	initSeeds []uint64,
	train, val []Sample,
	resumeFrom []uint64,
) (*dnaTrainResult, error) {
	printSampleFountainEquation()

	bursts := 4
	shards := 12
	shardN := 256
	anchorN := 64 // global mix into each shard fitness
	microGens := 6
	microClusters := 3
	microPer := 4
	topK := 4 // soft-gated ensemble
	lossRate := 0.30

	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		bursts = 2
		shards = 6
		shardN = 128
		anchorN = 32
		microGens = 3
		microClusters = 2
		microPer = 3
		topK = 3
	}
	if v := envInt("LOOM_SEED_MNIST_BURSTS"); v > 0 {
		bursts = v
	}
	if v := envInt("LOOM_SEED_MNIST_SHARDS"); v > 0 {
		shards = v
	}
	if v := envInt("LOOM_SEED_MNIST_SHARD_N"); v > 0 {
		shardN = v
	}
	if v := envInt("LOOM_SEED_MNIST_GEN"); v > 0 {
		microGens = maxInt(2, v/5)
	}
	if v := envInt("LOOM_SEED_MNIST_TOPK"); v > 0 {
		topK = v
	}

	rng := poly.NewSeedRNG(poly.SeedFrom("loom-seed-mnist-sample-fountain", initSeeds[0]))
	seeds := append([]uint64(nil), initSeeds...)
	if len(resumeFrom) == len(initSeeds) {
		seeds = append([]uint64(nil), resumeFrom...)
		fmt.Println("  ▶ sample-fountain resuming from saved seeds")
	}

	hofNet, err := rebuildNet(topo, sizes, dtypes, seeds)
	if err != nil {
		return nil, err
	}
	hofVal := evalAccuracy(hofNet, val)
	fmt.Printf("\n── Sample-fountain · start hof val=%.2f%% · bursts=%d · shards=%d×%d ──\n",
		hofVal*100, bursts, shards, shardN)
	printDeviationHeatmap(hofNet, val, min(2000, len(val)), "val BEFORE (sample-fountain)")

	cfg := defaultDNAPopConfig()
	cfg.Generations = microGens
	cfg.Clusters = microClusters
	cfg.PerCluster = microPer
	cfg.FitnessBatch = min(shardN+anchorN, cfg.FitnessBatch)
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		cfg.HeatmapVal = 500
	}

	globalGen := 0
	var cargos [][]byte
	bestFountainVal := 0.0
	freeByBurst := [][]int{{0}, {0, 1}, {0, 1, 2}, {0, 1, 2}}

	for burst := 1; burst <= bursts; burst++ {
		free := freeByBurst[(burst-1)%len(freeByBurst)]
		if len(free) > len(seeds) {
			free = free[:len(seeds)]
		}
		fmt.Printf("\n╔══ BURST %d/%d · sample shards · free=%v ══╗\n", burst, bursts, free)

		parts := partitionSampleShards(train, shards, shardN, rng)
		specialists := make([][]uint64, 0, len(parts))
		type ranked struct {
			seeds []uint64
			soft  float64
			full  float64
		}
		var elite []ranked

		for si, shard := range parts {
			fit := mixShardFitness(shard, train, anchorN, rng)
			fmt.Printf("\n── μ shard %d/%d · fit_n=%d · %d gens · free%v ──\n",
				si+1, len(parts), len(fit), cfg.Generations, free)

			hofBest, searchVal, packBest, gens, err := dnaSearchFreeSet(
				root, topo, sizes, dtypes, seeds, free, fit, val, cfg, rng, &globalGen,
			)
			if err != nil {
				return nil, err
			}
			// Pack soft specialist (may differ from full-val HOF) — this was the mode-6 bug.
			pack := packBest
			if len(pack) == 0 {
				pack = hofBest
			}
			cnet, err := rebuildNet(topo, sizes, dtypes, pack)
			if err != nil {
				return nil, err
			}
			fullVal := evalAccuracy(cnet, val)
			shardSoft := softFitness(cnet, fit)
			shardAcc := evalAccuracy(cnet, shard)

			// Always keep soft specialist for fountain pack (independent).
			specialists = append(specialists, append([]uint64(nil), pack...))
			elite = append(elite, ranked{seeds: pack, soft: shardSoft, full: fullVal})

			// Seed HOF uses full-val champion from this search (may be base if no lift).
			if searchVal > hofVal+1e-6 {
				seeds = append([]uint64(nil), hofBest...)
				hofVal = searchVal
				hofNet, err = rebuildNet(topo, sizes, dtypes, seeds)
				if err != nil {
					return nil, err
				}
				fmt.Printf("  ★ shard %d UPDATED seed HOF val=%.2f%%  (packSoft=%.4f packShardAcc=%.1f%% gens=%d)\n",
					si+1, hofVal*100, shardSoft, shardAcc*100, gens)
			} else {
				same := seedsEqual(pack, seeds)
				tag := "pack≠hof"
				if same {
					tag = "pack=hof (soft stuck)"
				}
				fmt.Printf("  ◆ shard %d pack-keep  packFull=%.2f%% hof=%.2f%% shardAcc=%.1f%% soft=%.4f  %s\n",
					si+1, fullVal*100, hofVal*100, shardAcc*100, shardSoft, tag)
			}
		}

		// Seed-side: pairwise pull among top elites by shard soft (not majority of all).
		if len(elite) >= 2 {
			sort.Slice(elite, func(i, j int) bool { return elite[i].soft < elite[j].soft })
			top := elite[:min(4, len(elite))]
			pulled := append([]uint64(nil), top[0].seeds...)
			for i := 1; i < len(top); i++ {
				pulled = pullSeedsToward(pulled, top[i].seeds, 0.35, 0.08, rng)
			}
			pNet, err := rebuildNet(topo, sizes, dtypes, pulled)
			if err != nil {
				return nil, err
			}
			pVal := evalAccuracy(pNet, val)
			fmt.Printf("\n── seed pairwise-pull (top-%d soft elites) val=%.2f%% ──\n", len(top), pVal*100)
			if pVal > hofVal+1e-6 {
				seeds = pulled
				hofVal = pVal
				hofNet = pNet
				fmt.Printf("  ★ pairwise UPDATED hof val=%.2f%%\n", hofVal*100)
			} else {
				fmt.Printf("  · pairwise held (%.2f%% ≤ hof %.2f%%)\n", pVal*100, hofVal*100)
			}
		}

		// κ) L1 fountain over independent specialists
		if len(specialists) < 2 {
			fmt.Println("  · L1 fountain skipped (need ≥2 specialists)")
			continue
		}
		blobs := make([][]byte, 0, len(specialists))
		origLens := make([]int, 0, len(specialists))
		for i, rs := range specialists {
			net, err := rebuildNet(topo, sizes, dtypes, rs)
			if err != nil {
				return nil, err
			}
			b, err := poly.PackNetworkWeights(net)
			if err != nil {
				return nil, fmt.Errorf("pack shard %d: %w", i, err)
			}
			blobs = append(blobs, b)
			origLens = append(origLens, len(b))
		}
		padded := padBlobsEqual(blobs)
		fseed := poly.SeedFrom("loom-seed-mnist-sample-l1", uint64(burst), uint64(len(padded)), uint64(len(padded[0])))
		fmt.Printf("\n── κ L1 fountain · K=%d · block=%dB · loss=%.0f%% ──\n",
			len(padded), len(padded[0]), lossRate*100)
		recovered, recv, sprayed, err := poly.RecoverWeightBlobs(padded, fseed, lossRate, 5.0)
		if err != nil {
			return nil, fmt.Errorf("L1 fountain: %w", err)
		}
		fmt.Printf("  recovered %d/%d  recv=%d  sprayed=%d\n", len(recovered), len(padded), recv, sprayed)

		experts := make([]*poly.VolumetricNetwork, len(recovered))
		for i, blob := range recovered {
			net, err := rebuildNet(topo, sizes, dtypes, seeds)
			if err != nil {
				return nil, err
			}
			use := blob
			if i < len(origLens) && len(use) > origLens[i] {
				use = use[:origLens[i]]
			}
			if err := poly.UnpackNetworkWeights(net, use); err != nil {
				return nil, fmt.Errorf("unpack L1 expert %d: %w", i, err)
			}
			experts[i] = net
		}
		master := &poly.FountainMaster{Experts: experts, K: len(experts), Recovered: len(experts), Received: recv, Sprayed: sprayed}
		fVal := evalMasterAcc(master, val)
		gateVal := evalMasterAccTopK(master, val, sampleSubset(val, min(512, len(val)), rng), topK)
		fmt.Printf("  ensemble val=%.2f%%  top-%d-gated=%.2f%%  (seed hof=%.2f%%)\n",
			fVal*100, topK, gateVal*100, hofVal*100)
		bestBurstF := math.Max(fVal, gateVal)
		if bestBurstF > bestFountainVal {
			bestFountainVal = bestBurstF
			fmt.Printf("  ★ fountain HOF %.2f%%\n", bestFountainVal*100)
		}

		cargo, err := packExpertsCargo(experts)
		if err != nil {
			return nil, err
		}
		cargos = append(cargos, cargo)
		fmt.Printf("  stashed L1 cargo %dB (cargos=%d)\n", len(cargo), len(cargos))

		// Warmth polish on seed HOF between bursts
		fmt.Printf("\n── polish · warmth on seed HOF ──\n")
		wcfg := defaultTrainConfig()
		wcfg.Epochs = maxInt(4, wcfg.Epochs/4)
		wcfg.MutPerLayer = maxInt(24, wcfg.MutPerLayer/2)
		if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
			wcfg.Epochs = 3
			wcfg.MutPerLayer = 16
		}
		before := hofVal
		snap := append([]uint64(nil), seeds...)
		trainLayerSeeds(hofNet, seeds, sizes, train, val, wcfg)
		after := evalAccuracy(hofNet, val)
		if after > before+1e-6 {
			hofVal = after
			fmt.Printf("  ★ warmth lifted hof %.2f%% → %.2f%%\n", before*100, hofVal*100)
		} else {
			fmt.Printf("  · warmth REJECTED after=%.2f%% (soft≠full-val trap) — restore seed hof %.2f%%\n",
				after*100, before*100)
			copy(seeds, snap)
			hofNet, err = rebuildNet(topo, sizes, dtypes, seeds)
			if err != nil {
				return nil, err
			}
			hofVal = before
		}

		_ = savePopCheckpoint(filepath.Join(root, popCheckpointFile), PopCheckpoint{
			Format:     popFormat,
			Generation: globalGen,
			Clusters:   cfg.Clusters,
			PerCluster: cfg.PerCluster,
			HofSeeds:   seeds,
			HofValAcc:  hofVal,
		})
	}

	if len(cargos) >= 2 {
		fmt.Printf("\n╔══ MEGA FOUNTAIN — consolidate consolidations (%d L1 cargos) ══╗\n", len(cargos))
		padded := padBlobsEqual(cargos)
		mseed := poly.SeedFrom("loom-seed-mnist-sample-mega", uint64(len(padded)), uint64(len(padded[0])))
		fmt.Printf("  padded %dB × K=%d  loss=%.0f%%\n", len(padded[0]), len(padded), lossRate*100)
		recovered, recv, sprayed, err := poly.RecoverWeightBlobs(padded, mseed, lossRate, 8.0)
		if err != nil {
			return nil, fmt.Errorf("mega fountain: %w", err)
		}
		fmt.Printf("  mega recovered %d/%d  recv=%d  sprayed=%d\n", len(recovered), len(padded), recv, sprayed)

		var megaExperts []*poly.VolumetricNetwork
		for ci, cargoBlob := range recovered {
			expertBlobs, err := unpackExpertsCargo(cargoBlob)
			if err != nil {
				fmt.Printf("  · cargo[%d] unpack failed: %v\n", ci, err)
				continue
			}
			for _, eb := range expertBlobs {
				net, err := rebuildNet(topo, sizes, dtypes, seeds)
				if err != nil {
					return nil, err
				}
				use := trimPackedOrRaw(eb, net)
				if err := poly.UnpackNetworkWeights(net, use); err != nil {
					ref, _ := poly.PackNetworkWeights(hofNet)
					if ref != nil && len(eb) >= len(ref) {
						if err2 := poly.UnpackNetworkWeights(net, eb[:len(ref)]); err2 != nil {
							continue
						}
					} else {
						continue
					}
				}
				megaExperts = append(megaExperts, net)
			}
		}
		if len(megaExperts) > 0 {
			mega := &poly.FountainMaster{Experts: megaExperts, K: len(megaExperts)}
			probe := sampleSubset(val, min(4000, len(val)), rng)
			mVal := evalMasterAcc(mega, probe)
			mGate := evalMasterAccTopK(mega, probe, sampleSubset(probe, min(512, len(probe)), rng), topK)
			fmt.Printf("  mega ensemble≈%.2f%%  top-%d-gated≈%.2f%%  (on %d val)  seed hof=%.2f%%  best L1=%.2f%%\n",
				mVal*100, topK, mGate*100, len(probe), hofVal*100, bestFountainVal*100)
			bestFountainVal = math.Max(bestFountainVal, math.Max(mVal, mGate))
		}
	} else {
		fmt.Println("\n── MEGA skipped (need ≥2 burst cargos) ──")
	}

	printDeviationHeatmap(hofNet, val, min(cfg.HeatmapVal, len(val)), "val AFTER sample-fountain (seed hof)")
	fmt.Printf("\n  ▸ sample-fountain done  seed_train=%.2f%%  seed_val=%.2f%%  fountain_best=%.2f%%  gens≈%d\n",
		evalAccuracy(hofNet, train)*100, hofVal*100, bestFountainVal*100, globalGen)

	return &dnaTrainResult{
		Seeds:      seeds,
		Net:        hofNet,
		Generation: globalGen,
		Clusters:   cfg.Clusters,
	}, nil
}

// partitionSampleShards draws K shards of up to n samples (reshuffle first).
func partitionSampleShards(train []Sample, k, n int, rng *poly.SeedRNG) [][]Sample {
	if k < 2 {
		k = 2
	}
	if n < 16 {
		n = 16
	}
	idx := make([]int, len(train))
	for i := range idx {
		idx[i] = i
	}
	for i := len(idx) - 1; i > 0; i-- {
		j := int(rng.Uint64() % uint64(i+1))
		idx[i], idx[j] = idx[j], idx[i]
	}
	out := make([][]Sample, k)
	cursor := 0
	for s := 0; s < k; s++ {
		need := n
		if cursor+need > len(idx) {
			cursor = 0
			for i := len(idx) - 1; i > 0; i-- {
				j := int(rng.Uint64() % uint64(i+1))
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
		shard := make([]Sample, need)
		for i := 0; i < need; i++ {
			shard[i] = train[idx[cursor+i]]
		}
		cursor += need
		out[s] = shard
	}
	return out
}

// mixShardFitness: shard samples + random global anchors (keeps specialists slightly global).
func mixShardFitness(shard, train []Sample, anchorN int, rng *poly.SeedRNG) []Sample {
	out := append([]Sample(nil), shard...)
	if anchorN <= 0 || len(train) == 0 {
		return out
	}
	anchors := sampleSubset(train, min(anchorN, len(train)), rng)
	out = append(out, anchors...)
	shuffleSamples(out, rng)
	return out
}

// evalMasterAccTopK averages only the k experts with best (lowest) soft fitness on probe.
func evalMasterAccTopK(m *poly.FountainMaster, samples, probe []Sample, k int) float64 {
	if m == nil || len(m.Experts) == 0 || len(samples) == 0 {
		return 0
	}
	type es struct {
		i    int
		soft float64
	}
	ranked := make([]es, 0, len(m.Experts))
	for i, e := range m.Experts {
		if e == nil {
			continue
		}
		ranked = append(ranked, es{i: i, soft: softFitness(e, probe)})
	}
	if len(ranked) == 0 {
		return 0
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].soft < ranked[b].soft })
	if k < 1 {
		k = 1
	}
	if k > len(ranked) {
		k = len(ranked)
	}
	picked := make([]*poly.VolumetricNetwork, 0, k)
	for i := 0; i < k; i++ {
		picked = append(picked, m.Experts[ranked[i].i])
	}
	gated := &poly.FountainMaster{Experts: picked, K: len(picked)}
	return evalMasterAcc(gated, samples)
}

func seedsEqual(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
