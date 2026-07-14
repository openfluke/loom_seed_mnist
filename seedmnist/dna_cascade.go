package seedmnist

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

/*
DNA cascade (mode 4) — aimed at beating coordinate / all-at-once DNA:

  Phase A  L0-only DNA islands (heavy budget) — first layer is the proven mover
  Phase B  Expand free-set {0,1} → {0,1,2} with LayerOverlaps-guided pull
  Phase C  Warmth bit-polish on hall-of-fame seeds

Keeps the seed manifold: mutate layer_seed → HeInit only.
Uses poly.CompareNetworks.LayerOverlaps so attraction μ/α track per-layer DNA
distance (OverallOverlap is inflated when frozen layers match).
*/

func printDNACascadeEquation() {
	fmt.Println(`
── DNA cascade (mode 4) — L0-heavy → expanding free-set → warmth polish ──
  A) DNA islands search ONLY s₀  (freeze s₁,s₂) · more gens
  B) free-set expands {s₀,s₁} then {s₀,s₁,s₂}
     α/μ from LayerOverlaps (not OverallOverlap) per free layer
  C) warmth-bit polish on hall-of-fame
  hof = best full val accuracy`)
}

func trainDNACascade(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	initSeeds []uint64,
	train, val []Sample,
	resumeFrom []uint64,
) (*dnaTrainResult, error) {
	printDNACascadeEquation()

	cfg := defaultDNAPopConfig()
	// Cascade gets a slightly leaner mid-phase so L0 budget stays fat.
	l0Cfg := cfg
	midCfg := cfg
	l0Cfg.Generations = cfg.Generations + cfg.Generations/2 // +50% on L0
	midCfg.Generations = maxInt(6, cfg.Generations*2/3)
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		l0Cfg.Generations = 8
		midCfg.Generations = 6
	}
	if v := envInt("LOOM_SEED_MNIST_GEN"); v > 0 {
		l0Cfg.Generations = v + v/2
		midCfg.Generations = maxInt(4, v*2/3)
	}

	rng := poly.NewSeedRNG(poly.SeedFrom("loom-seed-mnist-dna-cascade", initSeeds[0]))
	seeds := append([]uint64(nil), initSeeds...)
	if len(resumeFrom) == len(initSeeds) {
		seeds = append([]uint64(nil), resumeFrom...)
		fmt.Println("  ▶ cascade resuming from saved seeds")
	}

	hofNet, err := rebuildNet(topo, sizes, dtypes, seeds)
	if err != nil {
		return nil, err
	}
	hofVal := evalAccuracy(hofNet, val)
	fmt.Printf("\n── DNA cascade · start hof val=%.2f%% ──\n", hofVal*100)
	printDeviationHeatmap(hofNet, val, min(l0Cfg.HeatmapVal, len(val)), "val BEFORE (cascade)")

	globalGen := 0

	// ── Phase A: L0 only ──
	fmt.Printf("\n══ Phase A · focus L0 only · %d gens · %d×%d islands ══\n",
		l0Cfg.Generations, l0Cfg.Clusters, l0Cfg.PerCluster)
	bestL0, focusVal, gens, err := dnaSearchOneLayer(
		root, topo, sizes, dtypes, seeds, 0, train, val, l0Cfg, rng, &globalGen,
	)
	if err != nil {
		return nil, err
	}
	if focusVal > hofVal+1e-6 || (math.Abs(focusVal-hofVal) < 1e-6 && bestL0 != seeds[0]) {
		seeds[0] = bestL0
		hofVal = focusVal
		hofNet, err = rebuildNet(topo, sizes, dtypes, seeds)
		if err != nil {
			return nil, err
		}
		fmt.Printf("  ★ Phase A kept L0  seed=0x%x  hof val=%.2f%%  (%d gens)\n",
			seeds[0], hofVal*100, gens)
	} else {
		fmt.Printf("  · Phase A held L0  (best %.2f%% ≤ hof %.2f%%)\n", focusVal*100, hofVal*100)
	}
	printDeviationHeatmap(hofNet, val, min(l0Cfg.HeatmapVal, len(val)), "val after Phase A (L0)")

	// ── Phase B: expanding free-sets ──
	freeSets := [][]int{{0, 1}, {0, 1, 2}}
	for _, free := range freeSets {
		if len(free) > len(seeds) {
			continue
		}
		fmt.Printf("\n══ Phase B · free layers %v · %d gens ══\n", free, midCfg.Generations)
		best, v, _, g, err := dnaSearchFreeSet(
			root, topo, sizes, dtypes, seeds, free, train, val, midCfg, rng, &globalGen,
		)
		if err != nil {
			return nil, err
		}
		if v > hofVal+1e-6 {
			seeds = best
			hofVal = v
			hofNet, err = rebuildNet(topo, sizes, dtypes, seeds)
			if err != nil {
				return nil, err
			}
			fmt.Printf("  ★ Phase B UPDATED free=%v  hof val=%.2f%%  (%d gens)\n", free, hofVal*100, g)
		} else {
			fmt.Printf("  · Phase B held free=%v  (best %.2f%% ≤ hof %.2f%%)\n", free, v*100, hofVal*100)
		}
		printDeviationHeatmap(hofNet, val, min(midCfg.HeatmapVal, len(val)),
			fmt.Sprintf("val after Phase B free=%v", free))
	}

	// ── Phase C: warmth polish ──
	fmt.Printf("\n══ Phase C · warmth polish on hof ══\n")
	wcfg := defaultTrainConfig()
	wcfg.Epochs = maxInt(8, wcfg.Epochs/3)
	wcfg.MutPerLayer = maxInt(40, wcfg.MutPerLayer)
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		wcfg.Epochs = 6
		wcfg.MutPerLayer = 24
	}
	before := hofVal
	seedsSnap := append([]uint64(nil), seeds...)
	trainLayerSeeds(hofNet, seeds, sizes, train, val, wcfg)
	after := evalAccuracy(hofNet, val)
	if after > before+1e-6 {
		hofVal = after
		fmt.Printf("  ★ Phase C warmth lifted hof val %.2f%% → %.2f%%\n", before*100, hofVal*100)
	} else {
		fmt.Printf("  · Phase C warmth val %.2f%% (no beat of %.2f%%) — restore Phase B seeds\n",
			after*100, before*100)
		copy(seeds, seedsSnap)
		hofNet, err = rebuildNet(topo, sizes, dtypes, seeds)
		if err != nil {
			return nil, err
		}
		hofVal = before
	}

	printDeviationHeatmap(hofNet, val, min(l0Cfg.HeatmapVal, len(val)), "val AFTER cascade")
	fmt.Printf("\n  ▸ cascade done  train=%.2f%%  val=%.2f%%  gens≈%d\n",
		evalAccuracy(hofNet, train)*100, hofVal*100, globalGen)

	return &dnaTrainResult{
		Seeds:      seeds,
		Net:        hofNet,
		Generation: globalGen,
		Clusters:   midCfg.Clusters,
	}, nil
}

// dnaSearchFreeSet: DNA islands mutating only free layer indices; LayerOverlaps guide α/μ.
func dnaSearchFreeSet(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	baseSeeds []uint64,
	free []int,
	train, val []Sample,
	cfg DNAPopConfig,
	rng *poly.SeedRNG,
	globalGen *int,
) (bestSeeds []uint64, bestVal float64, packSeeds []uint64, gensRun int, err error) {
	fitness := sampleSubset(train, cfg.FitnessBatch, rng)
	base := append([]uint64(nil), baseSeeds...)
	bestSeeds = append([]uint64(nil), base...)
	packSeeds = append([]uint64(nil), base...)

	baseNet, err := rebuildNet(topo, sizes, dtypes, base)
	if err != nil {
		return nil, 0, nil, 0, err
	}
	bestVal = evalAccuracy(baseNet, val)
	bestSoft := softFitness(baseNet, fitness)
	packSoft := bestSoft
	freeSet := map[int]bool{}
	for _, li := range free {
		freeSet[li] = true
	}

	pop := make([]dnaGenome, 0, cfg.Clusters*cfg.PerCluster)
	for c := 0; c < cfg.Clusters; c++ {
		for i := 0; i < cfg.PerCluster; i++ {
			s := append([]uint64(nil), base...)
			if c > 0 || i > 0 {
				for _, li := range free {
					s[li] = mutateSeed(s[li], rng.Uint64())
					if c > 0 && i == 0 {
						s[li] = rng.Uint64()
					}
				}
			}
			pop = append(pop, dnaGenome{seeds: s, cluster: c})
		}
	}
	if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
		return nil, 0, nil, 0, err
	}
	// Latch soft specialist immediately if a mutant already beats base on this fitness set.
	{
		bi := bestIndex(pop)
		cand := freezeOutsideFreeCopy(pop[bi].seeds, base, freeSet)
		if pop[bi].fit < packSoft-1e-12 {
			packSoft = pop[bi].fit
			packSeeds = cand
		}
	}

	for gen := 1; gen <= cfg.Generations; gen++ {
		*globalGen++
		gensRun++
		fitness = sampleSubset(train, cfg.FitnessBatch, rng)
		if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
			return nil, 0, nil, 0, err
		}

		next := make([]dnaGenome, 0, len(pop))
		immigrants := 0
		for c := 0; c < cfg.Clusters; c++ {
			members := genomesInCluster(pop, c)
			sortGenomes(members)
			if len(members) == 0 {
				continue
			}
			collapse := clusterCollapse(members, topo, sizes, dtypes)
			eliteN := int(math.Ceil(float64(len(members)) * cfg.EliteFrac))
			if eliteN < 1 {
				eliteN = 1
			}
			if eliteN > len(members)/2 {
				eliteN = maxInt(1, len(members)/2)
			}
			elite := members[0]

			for i := 0; i < eliteN; i++ {
				g := cloneGenome(members[i])
				g.cluster = c
				freezeOutsideFree(g.seeds, base, freeSet)
				next = append(next, g)
			}

			need := cfg.PerCluster - eliteN
			immN := 0
			if collapse > 0.85 {
				immN = maxInt(1, int(math.Ceil(float64(need)*cfg.ImmigrantFrac)))
				if immN > need {
					immN = need
				}
			}

			eliteDNA, _ := genomeDNA(topo, sizes, dtypes, freezeOutsideFreeCopy(elite.seeds, base, freeSet))

			for i := 0; i < need; i++ {
				child := append([]uint64(nil), base...)
				if i < immN {
					for _, li := range free {
						child[li] = rng.Uint64()
					}
					immigrants++
				} else {
					a := members[int(rng.Uint64()%uint64(eliteN))]
					b := members[int(rng.Uint64()%uint64(eliteN))]
					for _, li := range free {
						mask := rng.Uint64()
						child[li] = (a.seeds[li] & mask) | (b.seeds[li] & ^mask)
					}
					childDNA, err := genomeDNA(topo, sizes, dtypes, child)
					if err != nil {
						return nil, 0, nil, 0, err
					}
					cmp := poly.CompareNetworks(childDNA, eliteDNA)
					gap := math.Max(0, softFitnessMust(topo, sizes, dtypes, child, fitness)-elite.fit)
					gapNorm := gap / (elite.fit + 1e-6)

					for _, li := range free {
						lov := layerOverlapAt(cmp, li)
						if lov < 0 {
							lov = 0
						}
						alpha := clip01(0.12 + 0.50*gapNorm + 0.28*(1-lov))
						mu := clip01(0.04 + 0.22*(1-lov) + 0.22*collapse)
						if mu < 0.02 {
							mu = 0.02
						}
						if mu > 0.45 {
							mu = 0.45
						}
						if collapse > 0.9 {
							alpha *= 0.35
						}
						pulled := pullSeedsToward(
							[]uint64{child[li]},
							[]uint64{elite.seeds[li]},
							alpha, mu, rng,
						)
						child[li] = pulled[0]
					}
				}
				freezeOutsideFree(child, base, freeSet)
				cnet, err := rebuildNet(topo, sizes, dtypes, child)
				if err != nil {
					return nil, 0, nil, 0, err
				}
				next = append(next, dnaGenome{
					seeds:   child,
					fit:     softFitness(cnet, fitness),
					acc:     evalAccuracy(cnet, fitness),
					cluster: c,
				})
			}
		}
		pop = next
		if cfg.MigrateEvery > 0 && gen%cfg.MigrateEvery == 0 && cfg.Clusters > 1 {
			migrateClustersFree(pop, cfg.Clusters, free, base, rng)
		}
		if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
			return nil, 0, nil, 0, err
		}

		bi := bestIndex(pop)
		cand := freezeOutsideFreeCopy(pop[bi].seeds, base, freeSet)
		candNet, err := rebuildNet(topo, sizes, dtypes, cand)
		if err != nil {
			return nil, 0, nil, 0, err
		}
		candVal := evalAccuracy(candNet, val)
		candSoft := softFitness(candNet, fitness)
		if candVal > bestVal || (math.Abs(candVal-bestVal) < 1e-6 && candSoft < bestSoft) {
			bestVal = candVal
			bestSoft = candSoft
			bestSeeds = cand
			fmt.Printf("  ★ free%v gen %d  val=%.2f%% soft=%.4f\n", free, gen, bestVal*100, bestSoft)
		}
		// Soft specialist for packing (mode 6): may diverge from full-val HOF.
		if candSoft < packSoft-1e-12 {
			packSoft = candSoft
			packSeeds = cand
			if math.Abs(candVal-bestVal) > 1e-6 {
				fmt.Printf("  ◆ pack soft↑ gen %d  soft=%.4f fullVal=%.2f%% (hofVal=%.2f%%)\n",
					gen, packSoft, candVal*100, bestVal*100)
			}
		}

		fmt.Printf("  free%v gen %02d/%d  bestSoft=%.4f batchAcc=%.1f%% focusVal=%.2f%% packSoft=%.4f imm=%d\n",
			free, gen, cfg.Generations, pop[bi].fit, pop[bi].acc*100, bestVal*100, packSoft, immigrants)

		_ = savePopCheckpoint(filepath.Join(root, popCheckpointFile), PopCheckpoint{
			Format:     popFormat,
			Generation: *globalGen,
			Clusters:   cfg.Clusters,
			PerCluster: cfg.PerCluster,
			HofSeeds:   bestSeeds,
			HofValAcc:  bestVal,
			Genomes:    serializePop(pop),
		})
	}
	return bestSeeds, bestVal, packSeeds, gensRun, nil
}

func layerOverlapAt(cmp poly.NetworkComparisonResult, li int) float64 {
	key := fmt.Sprintf("0,0,0,%d", li)
	if v, ok := cmp.LayerOverlaps[key]; ok {
		return float64(v)
	}
	return float64(cmp.OverallOverlap)
}

func freezeOutsideFree(seeds, base []uint64, free map[int]bool) {
	for i := range seeds {
		if !free[i] && i < len(base) {
			seeds[i] = base[i]
		}
	}
}

func freezeOutsideFreeCopy(seeds, base []uint64, free map[int]bool) []uint64 {
	out := append([]uint64(nil), base...)
	for i := range seeds {
		if free[i] && i < len(out) {
			out[i] = seeds[i]
		}
	}
	return out
}

func softFitnessMust(topo uint64, sizes []int, dtypes []string, seeds []uint64, fitness []Sample) float64 {
	net, err := rebuildNet(topo, sizes, dtypes, seeds)
	if err != nil {
		return 1e9
	}
	return softFitness(net, fitness)
}

func migrateClustersFree(pop []dnaGenome, clusters int, free []int, base []uint64, rng *poly.SeedRNG) {
	freeSet := map[int]bool{}
	for _, li := range free {
		freeSet[li] = true
	}
	for c := 0; c < clusters; c++ {
		idxs := []int{}
		for i, g := range pop {
			if g.cluster == c {
				idxs = append(idxs, i)
			}
		}
		if len(idxs) < 2 {
			continue
		}
		worst := idxs[len(idxs)-1]
		pop[worst].cluster = (c + 1) % clusters
		pop[worst].seeds = append([]uint64(nil), base...)
		for _, li := range free {
			pop[worst].seeds[li] = mutateSeed(pop[worst].seeds[li], rng.Uint64())
		}
	}
}
