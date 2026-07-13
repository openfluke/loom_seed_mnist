package seedmnist

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

/*
DNA layer-at-a-time (coordinate DNA):

  Round r cycles layers 0 → 1 → … → L-1
  While focusing layer ℓ:
    freeze s_j for j≠ℓ
    K DNA clusters search ONLY s_ℓ
    update s_ℓ if full val acc improves
*/
func printDNALayerEquation() {
	fmt.Println(`
── DNA layer-at-a-time (coordinate DNA clusters) ──
  for each round · for each layer ℓ:
    freeze other layer seeds
    K clusters search ONLY s_ℓ  (DNA attract + immigrants)
    keep s_ℓ if full val acc improves`)
}

func defaultDNALayerConfig() DNAPopConfig {
	cfg := DNAPopConfig{
		Clusters:      3,
		PerCluster:    5,
		Generations:   8,
		FitnessBatch:  384,
		HeatmapVal:    2000,
		HeatmapEvery:  0, // heatmaps between layers/rounds in parent
		FullEvalEvery: 0,
		EliteFrac:     0.34,
		MigrateEvery:  2,
		ImmigrantFrac: 0.30,
	}
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		cfg.Clusters = 2
		cfg.PerCluster = 4
		cfg.Generations = 4
		cfg.FitnessBatch = 192
		cfg.HeatmapVal = 500
	}
	if v := envInt("LOOM_SEED_MNIST_CLUSTERS"); v > 0 {
		cfg.Clusters = v
	}
	if v := envInt("LOOM_SEED_MNIST_GEN"); v > 0 {
		cfg.Generations = v
	}
	return cfg
}

func trainDNALayerWise(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	initSeeds []uint64,
	train, val []Sample,
	resumeFrom []uint64,
) (*dnaTrainResult, error) {
	printDNALayerEquation()
	cfg := defaultDNALayerConfig()
	rng := poly.NewSeedRNG(poly.SeedFrom("loom-seed-mnist-dna-layer", initSeeds[0]))

	rounds := 3
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		rounds = 2
	}
	if v := envInt("LOOM_SEED_MNIST_ROUNDS"); v > 0 {
		rounds = v
	}

	seeds := append([]uint64(nil), initSeeds...)
	if len(resumeFrom) == len(initSeeds) {
		seeds = append([]uint64(nil), resumeFrom...)
		fmt.Println("  ▶ starting layer-wise DNA from resumed seeds")
	}

	hofNet, err := rebuildNet(topo, sizes, dtypes, seeds)
	if err != nil {
		return nil, err
	}
	hofVal := evalAccuracy(hofNet, val)
	fmt.Printf("\n── DNA layer-wise: %d rounds × %d layers · %d×%d clusters · %d gens/layer ──\n",
		rounds, len(seeds), cfg.Clusters, cfg.PerCluster, cfg.Generations)
	fmt.Printf("  start hof val=%.2f%%\n", hofVal*100)
	printDeviationHeatmap(hofNet, val, min(cfg.HeatmapVal, len(val)), "val BEFORE (dna-layer)")

	globalGen := 0
	nLayers := len(seeds)

	for round := 1; round <= rounds; round++ {
		fmt.Printf("\n══ Round %d/%d ══\n", round, rounds)
		for li := 0; li < nLayers; li++ {
			fmt.Printf("\n── Focus layer %d/%d (%dx%d) · others frozen ──\n",
				li, nLayers-1, sizes[li], sizes[li+1])

			bestLayerSeed, focusVal, gens, err := dnaSearchOneLayer(
				root, topo, sizes, dtypes, seeds, li, train, val, cfg, rng, &globalGen,
			)
			if err != nil {
				return nil, err
			}
			if focusVal > hofVal+1e-6 {
				seeds[li] = bestLayerSeed
				hofVal = focusVal
				hofNet, err = rebuildNet(topo, sizes, dtypes, seeds)
				if err != nil {
					return nil, err
				}
				fmt.Printf("  ★ layer %d UPDATED  seed=0x%x  hof val=%.2f%%  (%d gens)\n",
					li, seeds[li], hofVal*100, gens)
			} else if focusVal >= hofVal-1e-6 && bestLayerSeed != seeds[li] {
				// tie on val — accept if soft better later; keep seed change if equal val
				seeds[li] = bestLayerSeed
				hofNet, err = rebuildNet(topo, sizes, dtypes, seeds)
				if err != nil {
					return nil, err
				}
				fmt.Printf("  · layer %d seed swapped (val≈%.2f%%) seed=0x%x\n", li, hofVal*100, seeds[li])
			} else {
				fmt.Printf("  · layer %d held  (focus best val %.2f%% ≤ hof %.2f%%)\n",
					li, focusVal*100, hofVal*100)
			}
		}
		printDeviationHeatmap(hofNet, val, min(cfg.HeatmapVal, len(val)),
			fmt.Sprintf("val after round %d", round))
		fmt.Printf("  ▸ round %d hof  train=%.2f%% val=%.2f%%\n",
			round, evalAccuracy(hofNet, train)*100, hofVal*100)
	}

	return &dnaTrainResult{
		Seeds:      seeds,
		Net:        hofNet,
		Generation: globalGen,
		Clusters:   cfg.Clusters,
	}, nil
}

func dnaSearchOneLayer(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	baseSeeds []uint64,
	focus int,
	train, val []Sample,
	cfg DNAPopConfig,
	rng *poly.SeedRNG,
	globalGen *int,
) (bestSeed uint64, bestVal float64, gensRun int, err error) {
	fitness := sampleSubset(train, cfg.FitnessBatch, rng)
	base := append([]uint64(nil), baseSeeds...)
	bestSeed = base[focus]

	baseNet, err := rebuildNet(topo, sizes, dtypes, base)
	if err != nil {
		return 0, 0, 0, err
	}
	bestVal = evalAccuracy(baseNet, val)
	bestSoft := softFitness(baseNet, fitness)

	pop := make([]dnaGenome, 0, cfg.Clusters*cfg.PerCluster)
	for c := 0; c < cfg.Clusters; c++ {
		for i := 0; i < cfg.PerCluster; i++ {
			s := append([]uint64(nil), base...)
			if c > 0 || i > 0 {
				s[focus] = mutateSeed(s[focus], rng.Uint64())
				if c > 0 && i == 0 {
					s[focus] = rng.Uint64()
				} else if rng.Uint64()%2 == 0 {
					s[focus] = mutateSeed(s[focus], rng.Uint64())
				}
			}
			pop = append(pop, dnaGenome{seeds: s, cluster: c})
		}
	}
	if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
		return 0, 0, 0, err
	}

	for gen := 1; gen <= cfg.Generations; gen++ {
		*globalGen++
		gensRun++
		fitness = sampleSubset(train, cfg.FitnessBatch, rng)
		if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
			return 0, 0, 0, err
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
				for j := range g.seeds {
					if j != focus {
						g.seeds[j] = base[j]
					}
				}
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

			for i := 0; i < need; i++ {
				child := append([]uint64(nil), base...)
				if i < immN {
					child[focus] = rng.Uint64()
					immigrants++
				} else {
					a := members[int(rng.Uint64()%uint64(eliteN))]
					b := members[int(rng.Uint64()%uint64(eliteN))]
					mask := rng.Uint64()
					child[focus] = (a.seeds[focus] & mask) | (b.seeds[focus] & ^mask)

					childNet, err := rebuildNet(topo, sizes, dtypes, child)
					if err != nil {
						return 0, 0, 0, err
					}
					eliteSeeds := freezeExcept(elite.seeds, base, focus)
					eliteNet, err := rebuildNet(topo, sizes, dtypes, eliteSeeds)
					if err != nil {
						return 0, 0, 0, err
					}
					overlap := float64(poly.CompareNetworks(poly.ExtractDNA(childNet), poly.ExtractDNA(eliteNet)).OverallOverlap)
					gap := math.Max(0, softFitness(childNet, fitness)-elite.fit)
					gapNorm := gap / (elite.fit + 1e-6)
					alpha := clip01(0.10 + 0.55*gapNorm + 0.20*(1-overlap))
					mu := clip01(0.03 + 0.20*(1-overlap) + 0.25*collapse)
					if mu < 0.02 {
						mu = 0.02
					}
					if mu > 0.40 {
						mu = 0.40
					}
					if collapse > 0.9 {
						alpha *= 0.3
					}
					pulled := pullSeedsToward(
						[]uint64{child[focus]},
						[]uint64{elite.seeds[focus]},
						alpha, mu, rng,
					)
					child[focus] = pulled[0]
				}
				cnet, err := rebuildNet(topo, sizes, dtypes, child)
				if err != nil {
					return 0, 0, 0, err
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
			migrateClustersFocus(pop, cfg.Clusters, focus, base, rng)
		}
		if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
			return 0, 0, 0, err
		}

		bi := bestIndex(pop)
		cand := freezeExcept(pop[bi].seeds, base, focus)
		candNet, err := rebuildNet(topo, sizes, dtypes, cand)
		if err != nil {
			return 0, 0, 0, err
		}
		candVal := evalAccuracy(candNet, val)
		candSoft := softFitness(candNet, fitness)
		if candVal > bestVal || (math.Abs(candVal-bestVal) < 1e-6 && candSoft < bestSoft) {
			bestVal = candVal
			bestSoft = candSoft
			bestSeed = cand[focus]
			fmt.Printf("  ★ focus L%d gen %d  val=%.2f%% soft=%.4f seed=0x%x\n",
				focus, gen, bestVal*100, bestSoft, bestSeed)
		}

		col := 0.0
		nc := 0
		for c := 0; c < cfg.Clusters; c++ {
			m := genomesInCluster(pop, c)
			if len(m) >= 2 {
				col += clusterCollapse(m, topo, sizes, dtypes)
				nc++
			}
		}
		if nc > 0 {
			col /= float64(nc)
		}

		fmt.Printf("  L%d gen %02d/%d  bestSoft=%.4f batchAcc=%.1f%% focusVal=%.2f%% imm=%d collapse=%.2f\n",
			focus, gen, cfg.Generations, pop[bi].fit, pop[bi].acc*100, bestVal*100, immigrants, col)

		hof := append([]uint64(nil), base...)
		hof[focus] = bestSeed
		_ = savePopCheckpoint(filepath.Join(root, popCheckpointFile), PopCheckpoint{
			Format:     popFormat,
			Generation: *globalGen,
			Clusters:   cfg.Clusters,
			PerCluster: cfg.PerCluster,
			HofSeeds:   hof,
			HofValAcc:  bestVal,
			Genomes:    serializePop(pop),
		})
	}
	return bestSeed, bestVal, gensRun, nil
}

func freezeExcept(seeds, base []uint64, focus int) []uint64 {
	out := append([]uint64(nil), base...)
	if focus >= 0 && focus < len(seeds) && focus < len(out) {
		out[focus] = seeds[focus]
	}
	return out
}

func migrateClustersFocus(pop []dnaGenome, clusters, focus int, base []uint64, rng *poly.SeedRNG) {
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
		pop[worst].seeds[focus] = mutateSeed(pop[worst].seeds[focus], rng.Uint64())
	}
}
