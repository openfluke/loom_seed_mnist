package seedmnist

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/loom/poly"
)

// TrainMode selects the seed-search algorithm.
type TrainMode string

const (
	ModeWarmth   TrainMode = "warmth"    // single genome · warm-bit hill-climb
	ModeDNA      TrainMode = "dna"       // clustered multi-seed DNA attract (all layers)
	ModeDNALayer TrainMode = "dna-layer" // DNA clusters · one layer seed at a time
)

const popCheckpointFile = "mnist.pop.json"

func resolveTrainMode() TrainMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LOOM_SEED_MNIST_MODE"))) {
	case "dna-layer", "dnalayer", "layer", "coord", "coordinate":
		return ModeDNALayer
	case "dna", "neat", "pop", "population", "cluster":
		return ModeDNA
	case "warmth", "warm", "hill", "bits":
		return ModeWarmth
	default:
		return ModeWarmth
	}
}

// ResolveTrainMode is the exported mode picker used by main/run.sh.
func ResolveTrainMode() TrainMode { return resolveTrainMode() }

// SetTrainMode overrides mode (from menu / CLI) before training.
func SetTrainMode(m TrainMode) {
	switch m {
	case ModeDNA:
		_ = os.Setenv("LOOM_SEED_MNIST_MODE", "dna")
	case ModeDNALayer:
		_ = os.Setenv("LOOM_SEED_MNIST_MODE", "dna-layer")
	default:
		_ = os.Setenv("LOOM_SEED_MNIST_MODE", "warmth")
	}
}

// WantContinue reports LOOM_SEED_MNIST_CONTINUE=1 / continue flag.
func WantContinue() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("LOOM_SEED_MNIST_CONTINUE")))
	return v == "1" || v == "true" || v == "yes" || v == "continue"
}

// SetContinue toggles resume-from-saved-seeds.
func SetContinue(on bool) {
	if on {
		_ = os.Setenv("LOOM_SEED_MNIST_CONTINUE", "1")
	} else {
		_ = os.Unsetenv("LOOM_SEED_MNIST_CONTINUE")
	}
}

/*
Clustered DNA attract (seeds-only):

  K independent clusters explore in parallel.
  Each cluster has its own elite e*_k and local population.
  Periodically: migrate elites across clusters (island model).

  Per genome g in cluster k:
    W=HeInit(s) → DNA=Normalize(W) → F=softFitness
    overlap_local = DNA cosine vs e*_k
    gap = max(0, F(g)-F(e*_k)) / (F(e*_k)+ε)

    # ANTI-COLLAPSE: when cluster mean pairwise DNA cosine → 1, crank μ
    α = clip(0.10 + 0.55·gap + 0.20·(1-overlap), 0, 0.85)
    μ = clip(0.03 + 0.20·(1-overlap) + 0.25·collapse, 0.02, 0.35)

    s′ = s ⊕ ((s ⊕ e*_k) ∧ M_α) ⊕ M_μ
    + fresh immigrants when collapse > 0.9
    + hall-of-fame kept by full-val accuracy (stable, not batch soft)
*/
func printDNAAttractEquation() {
	fmt.Println(`
── Clustered DNA multi-seed attract (seeds-only) ──
  K clusters · each has elite e*_k · island migrate every few gens
  α = clip(0.10 + 0.55·gap + 0.20·(1-overlap), 0, 0.85)
  μ = clip(0.03 + 0.20·(1-overlap) + 0.25·collapse, 0.02, 0.35)
  s′ = s ⊕ ((s ⊕ e*) ∧ M_α) ⊕ M_μ
  collapse≈1 → inject immigrants · hall-of-fame by full val acc`)
}

type dnaGenome struct {
	seeds   []uint64
	fit     float64
	acc     float64
	cluster int
}

type DNAPopConfig struct {
	Clusters      int
	PerCluster    int
	Generations   int
	FitnessBatch  int
	HeatmapVal    int
	HeatmapEvery  int
	FullEvalEvery int
	EliteFrac     float64
	MigrateEvery  int
	ImmigrantFrac float64
}

func defaultDNAPopConfig() DNAPopConfig {
	cfg := DNAPopConfig{
		Clusters:      4,
		PerCluster:    6, // 24 total genomes
		Generations:   30,
		FitnessBatch:  384,
		HeatmapVal:    2000,
		HeatmapEvery:  5,
		FullEvalEvery: 5,
		EliteFrac:     0.34, // ~2 elites per cluster of 6
		MigrateEvery:  3,
		ImmigrantFrac: 0.25,
	}
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		cfg.Clusters = 3
		cfg.PerCluster = 4
		cfg.Generations = 12
		cfg.FitnessBatch = 192
		cfg.HeatmapVal = 500
		cfg.HeatmapEvery = 3
		cfg.FullEvalEvery = 4
		cfg.MigrateEvery = 2
	}
	if v := envInt("LOOM_SEED_MNIST_CLUSTERS"); v > 0 {
		cfg.Clusters = v
	}
	if v := envInt("LOOM_SEED_MNIST_PER_CLUSTER"); v > 0 {
		cfg.PerCluster = v
	}
	if v := envInt("LOOM_SEED_MNIST_POP"); v > 0 {
		// compatibility: total pop → redistribute
		cfg.PerCluster = maxInt(2, v/maxInt(1, cfg.Clusters))
	}
	if v := envInt("LOOM_SEED_MNIST_GEN"); v > 0 {
		cfg.Generations = v
	}
	return cfg
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// PopCheckpoint lets DNA search resume mid-exploration.
type PopCheckpoint struct {
	Format     string          `json:"format"`
	Generation int             `json:"generation"`
	Clusters   int             `json:"clusters"`
	PerCluster int             `json:"per_cluster"`
	HofSeeds   []uint64        `json:"hof_seeds"`
	HofValAcc  float64         `json:"hof_val_acc"`
	Genomes    []checkpointGen `json:"genomes"`
}

type checkpointGen struct {
	Cluster int      `json:"cluster"`
	Seeds   []uint64 `json:"seeds"`
}

const popFormat = "chaosglue-seed-mnist-pop-v2"

type dnaTrainResult struct {
	Seeds      []uint64
	Net        *poly.VolumetricNetwork
	Generation int
	Clusters   int
}

func trainDNAPopulation(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	initSeeds []uint64,
	train, val []Sample,
	cfg DNAPopConfig,
	resumeFrom []uint64,
) (*dnaTrainResult, error) {
	printDNAAttractEquation()
	rng := poly.NewSeedRNG(poly.SeedFrom("loom-seed-mnist-dna-v2", initSeeds[0], uint64(cfg.Clusters)))
	fitness := sampleSubset(train, cfg.FitnessBatch, rng)

	total := cfg.Clusters * cfg.PerCluster
	pop := make([]dnaGenome, 0, total)
	startGen := 0

	// Resume: prefer full population checkpoint; else seed file as hof + diversify
	cpPath := filepath.Join(root, popCheckpointFile)
	if WantContinue() {
		if cp, err := loadPopCheckpoint(cpPath); err == nil && len(cp.Genomes) > 0 {
			fmt.Printf("  ▶ CONTINUE from %s (gen %d, %d genomes)\n", popCheckpointFile, cp.Generation, len(cp.Genomes))
			startGen = cp.Generation
			if cp.Clusters > 0 {
				cfg.Clusters = cp.Clusters
			}
			if cp.PerCluster > 0 {
				cfg.PerCluster = cp.PerCluster
			}
			for _, g := range cp.Genomes {
				pop = append(pop, dnaGenome{seeds: append([]uint64(nil), g.Seeds...), cluster: g.Cluster})
			}
			for len(pop) < cfg.Clusters*cfg.PerCluster {
				c := len(pop) % cfg.Clusters
				s := diversifySeeds(cp.HofSeeds, rng, 4)
				pop = append(pop, dnaGenome{seeds: s, cluster: c})
			}
		} else if len(resumeFrom) == len(initSeeds) {
			fmt.Println("  ▶ CONTINUE from mnist.seeds.json (no pop checkpoint — seeding clusters around saved elite)")
			for c := 0; c < cfg.Clusters; c++ {
				for i := 0; i < cfg.PerCluster; i++ {
					s := append([]uint64(nil), resumeFrom...)
					if !(c == 0 && i == 0) {
						// each cluster gets a different neighborhood around the saved elite
						strength := 2 + c + i/2
						s = diversifySeeds(s, rng, strength)
					}
					pop = append(pop, dnaGenome{seeds: s, cluster: c})
				}
			}
		}
	}

	if len(pop) == 0 {
		// Fresh start: cluster 0 around topology init; other clusters random-jump neighborhoods
		base := initSeeds
		if len(resumeFrom) == len(initSeeds) && !WantContinue() {
			base = resumeFrom
		}
		for c := 0; c < cfg.Clusters; c++ {
			for i := 0; i < cfg.PerCluster; i++ {
				s := append([]uint64(nil), base...)
				if c > 0 || i > 0 {
					s = diversifySeeds(s, rng, 3+2*c+i)
				}
				if c > 0 && i == 0 {
					// cluster founder: far jump so DNA regions differ
					for li := range s {
						s[li] = rng.Uint64()
					}
				}
				pop = append(pop, dnaGenome{seeds: s, cluster: c})
			}
		}
	}

	if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
		return nil, err
	}

	hofSeeds := append([]uint64(nil), pop[bestIndex(pop)].seeds...)
	hofNet, err := rebuildNet(topo, sizes, dtypes, hofSeeds)
	if err != nil {
		return nil, err
	}
	hofVal := evalAccuracy(hofNet, val)
	startSoft := pop[bestIndex(pop)].fit

	fmt.Printf("\n── DNA clusters: K=%d × %d = %d genomes · gens=%d (from %d) · batch=%d ──\n",
		cfg.Clusters, cfg.PerCluster, len(pop), cfg.Generations, startGen, cfg.FitnessBatch)
	fmt.Printf("  start best soft=%.4f batch-acc=%.1f%%  hof val=%.2f%%\n",
		startSoft, pop[bestIndex(pop)].acc*100, hofVal*100)
	printClusterSummary(pop, topo, sizes, dtypes)
	printDeviationHeatmap(hofNet, val, min(cfg.HeatmapVal, len(val)), "val BEFORE (clustered DNA)")

	endGen := startGen + cfg.Generations
	for gen := startGen + 1; gen <= endGen; gen++ {
		fitness = sampleSubset(train, cfg.FitnessBatch, rng)
		if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
			return nil, err
		}

		next := make([]dnaGenome, 0, len(pop))
		immigrants := 0
		pulls := 0

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
			eliteDNA, err := genomeDNA(topo, sizes, dtypes, elite.seeds)
			if err != nil {
				return nil, err
			}

			// elites survive
			for i := 0; i < eliteN; i++ {
				g := cloneGenome(members[i])
				g.cluster = c
				next = append(next, g)
			}

			need := cfg.PerCluster - eliteN
			immN := 0
			if collapse > 0.85 {
				immN = int(math.Ceil(float64(need) * cfg.ImmigrantFrac * (0.5 + collapse)))
				if immN < 1 {
					immN = 1
				}
				if immN > need {
					immN = need
				}
			}

			for i := 0; i < need; i++ {
				var childSeeds []uint64
				if i < immN {
					// fresh immigrant — new DNA island inside cluster
					childSeeds = diversifySeeds(elite.seeds, rng, 8+int(collapse*8))
					for li := range childSeeds {
						if rng.Uint64()%2 == 0 {
							childSeeds[li] = rng.Uint64()
						}
					}
					immigrants++
				} else {
					a := members[int(rng.Uint64()%uint64(eliteN))]
					b := members[int(rng.Uint64()%uint64(eliteN))]
					childSeeds = crossoverSeeds(a.seeds, b.seeds, rng)
					childNet, err := rebuildNet(topo, sizes, dtypes, childSeeds)
					if err != nil {
						return nil, err
					}
					overlap := float64(poly.CompareNetworks(poly.ExtractDNA(childNet), eliteDNA).OverallOverlap)
					gap := math.Max(0, softFitness(childNet, fitness)-elite.fit)
					gapNorm := gap / (elite.fit + 1e-6)
					alpha := clip01(0.10 + 0.55*gapNorm + 0.20*(1-overlap))
					mu := clip01(0.03 + 0.20*(1-overlap) + 0.25*collapse)
					if mu < 0.02 {
						mu = 0.02
					}
					if mu > 0.35 {
						mu = 0.35
					}
					// when collapsed, pull LESS (don't clone elite) and mutate MORE
					if collapse > 0.9 {
						alpha *= 0.35
					}
					childSeeds = pullSeedsToward(childSeeds, elite.seeds, alpha, mu, rng)
					pulls++
				}
				childNet, err := rebuildNet(topo, sizes, dtypes, childSeeds)
				if err != nil {
					return nil, err
				}
				next = append(next, dnaGenome{
					seeds:   childSeeds,
					fit:     softFitness(childNet, fitness),
					acc:     evalAccuracy(childNet, fitness),
					cluster: c,
				})
			}
		}
		pop = next

		// island migration: swap one explorer between neighboring clusters
		if cfg.MigrateEvery > 0 && gen%cfg.MigrateEvery == 0 && cfg.Clusters > 1 {
			migrateClusters(pop, cfg.Clusters, rng)
		}

		if err := scorePopulation(pop, topo, sizes, dtypes, fitness); err != nil {
			return nil, err
		}

		// update hall-of-fame using full val accuracy (stable signal)
		bi := bestIndex(pop)
		candNet, err := rebuildNet(topo, sizes, dtypes, pop[bi].seeds)
		if err != nil {
			return nil, err
		}
		candVal := evalAccuracy(candNet, val)
		if candVal > hofVal {
			hofVal = candVal
			hofSeeds = append([]uint64(nil), pop[bi].seeds...)
			hofNet = candNet
			fmt.Printf("  ★ new hall-of-fame val acc=%.2f%%\n", hofVal*100)
		}

		improved := (startSoft - pop[bi].fit) / math.Max(startSoft, 1e-6)
		if improved < 0 {
			improved = 0
		}
		if improved > 1 {
			improved = 1
		}
		filled := int(improved * 20)
		bar := strings.Repeat("█", filled) + strings.Repeat("░", 20-filled)

		fmt.Printf("  gen %02d/%d  best soft=%.4f batch-acc=%.1f%%  hof val=%.2f%%  pulls=%d imm=%d\n",
			gen, endGen, pop[bi].fit, pop[bi].acc*100, hofVal*100, pulls, immigrants)
		fmt.Printf("           [%s] soft↓%.1f%% vs start\n", bar, improved*100)
		printClusterSummary(pop, topo, sizes, dtypes)

		if err := savePopCheckpoint(cpPath, PopCheckpoint{
			Format:     popFormat,
			Generation: gen,
			Clusters:   cfg.Clusters,
			PerCluster: cfg.PerCluster,
			HofSeeds:   hofSeeds,
			HofValAcc:  hofVal,
			Genomes:    serializePop(pop),
		}); err != nil {
			fmt.Printf("  WARN checkpoint: %v\n", err)
		}

		if cfg.HeatmapEvery > 0 && gen%cfg.HeatmapEvery == 0 {
			printDeviationHeatmap(hofNet, val, min(cfg.HeatmapVal, len(val)),
				fmt.Sprintf("val hof gen %d", gen))
		}
		if cfg.FullEvalEvery > 0 && gen%cfg.FullEvalEvery == 0 {
			tr := evalAccuracy(hofNet, train)
			fmt.Printf("  ▸ full eval hof  train acc=%.2f%%  val acc=%.2f%%\n", tr*100, hofVal*100)
		}
	}

	hofNet, err = rebuildNet(topo, sizes, dtypes, hofSeeds)
	if err != nil {
		return nil, err
	}
	return &dnaTrainResult{
		Seeds:      hofSeeds,
		Net:        hofNet,
		Generation: endGen,
		Clusters:   cfg.Clusters,
	}, nil
}

func diversifySeeds(base []uint64, rng *poly.SeedRNG, strength int) []uint64 {
	out := append([]uint64(nil), base...)
	if strength < 1 {
		strength = 1
	}
	for i := 0; i < strength; i++ {
		li := int(rng.Uint64() % uint64(len(out)))
		out[li] = mutateSeed(out[li], rng.Uint64())
	}
	return out
}

func scorePopulation(pop []dnaGenome, topo uint64, sizes []int, dtypes []string, fitness []Sample) error {
	for i := range pop {
		net, err := rebuildNet(topo, sizes, dtypes, pop[i].seeds)
		if err != nil {
			return err
		}
		pop[i].fit = softFitness(net, fitness)
		pop[i].acc = evalAccuracy(net, fitness)
	}
	return nil
}

func genomesInCluster(pop []dnaGenome, c int) []dnaGenome {
	var out []dnaGenome
	for _, g := range pop {
		if g.cluster == c {
			out = append(out, cloneGenome(g))
		}
	}
	return out
}

func bestIndex(pop []dnaGenome) int {
	bi := 0
	for i := 1; i < len(pop); i++ {
		if pop[i].fit < pop[bi].fit {
			bi = i
		}
	}
	return bi
}

func clusterCollapse(members []dnaGenome, topo uint64, sizes []int, dtypes []string) float64 {
	if len(members) < 2 {
		return 0
	}
	dnas := make([]poly.NetworkDNA, len(members))
	for i := range members {
		d, err := genomeDNA(topo, sizes, dtypes, members[i].seeds)
		if err != nil {
			return 0
		}
		dnas[i] = d
	}
	var sum float64
	n := 0
	for i := 0; i < len(dnas); i++ {
		for j := i + 1; j < len(dnas); j++ {
			ov := float64(poly.CompareNetworks(dnas[i], dnas[j]).OverallOverlap)
			if ov < 0 {
				ov = 0
			}
			sum += ov
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func printClusterSummary(pop []dnaGenome, topo uint64, sizes []int, dtypes []string) {
	maxC := 0
	for _, g := range pop {
		if g.cluster > maxC {
			maxC = g.cluster
		}
	}
	for c := 0; c <= maxC; c++ {
		m := genomesInCluster(pop, c)
		if len(m) == 0 {
			continue
		}
		sortGenomes(m)
		col := clusterCollapse(m, topo, sizes, dtypes)
		pair := float32(0)
		if len(m) > 1 {
			d0, _ := genomeDNA(topo, sizes, dtypes, m[0].seeds)
			d1, _ := genomeDNA(topo, sizes, dtypes, m[1].seeds)
			pair = poly.CompareNetworks(d0, d1).OverallOverlap
		}
		fmt.Printf("           cluster %d  n=%d  elite soft=%.4f acc=%.1f%%  collapse=%.2f  #1↔#2=%.3f\n",
			c, len(m), m[0].fit, m[0].acc*100, col, pair)
	}
}

func migrateClusters(pop []dnaGenome, clusters int, rng *poly.SeedRNG) {
	// move worst member of cluster c into c+1 after heavy mutate
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
		// pick last in idxs as migratee (approximate — re-sort would be better)
		worst := idxs[len(idxs)-1]
		dest := (c + 1) % clusters
		pop[worst].cluster = dest
		pop[worst].seeds = diversifySeeds(pop[worst].seeds, rng, 5)
	}
}

func serializePop(pop []dnaGenome) []checkpointGen {
	out := make([]checkpointGen, len(pop))
	for i, g := range pop {
		out[i] = checkpointGen{Cluster: g.cluster, Seeds: append([]uint64(nil), g.seeds...)}
	}
	return out
}

func savePopCheckpoint(path string, cp PopCheckpoint) error {
	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadPopCheckpoint(path string) (PopCheckpoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PopCheckpoint{}, err
	}
	var cp PopCheckpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return PopCheckpoint{}, err
	}
	if cp.Format != popFormat {
		return PopCheckpoint{}, fmt.Errorf("unknown pop format %q", cp.Format)
	}
	return cp, nil
}

func genomeDNA(topo uint64, sizes []int, dtypes []string, seeds []uint64) (poly.NetworkDNA, error) {
	net, err := rebuildNet(topo, sizes, dtypes, seeds)
	if err != nil {
		return nil, err
	}
	return poly.ExtractDNA(net), nil
}

func pullSeedsToward(self, elite []uint64, alpha, mu float64, rng *poly.SeedRNG) []uint64 {
	out := make([]uint64, len(self))
	for i := range self {
		diff := self[i] ^ elite[i]
		var pullMask, mutMask uint64
		for b := 0; b < 64; b++ {
			bit := uint64(1) << b
			if diff&bit != 0 && bernoulli(rng, alpha) {
				pullMask |= bit
			}
			if bernoulli(rng, mu) {
				mutMask |= bit
			}
		}
		out[i] = self[i] ^ pullMask ^ mutMask
	}
	return out
}

func crossoverSeeds(a, b []uint64, rng *poly.SeedRNG) []uint64 {
	out := make([]uint64, len(a))
	for i := range a {
		mask := rng.Uint64()
		out[i] = (a[i] & mask) | (b[i] & ^mask)
	}
	return out
}

func bernoulli(rng *poly.SeedRNG, p float64) bool {
	if p <= 0 {
		return false
	}
	if p >= 1 {
		return true
	}
	return float64(rng.Uint64()%1000000)/1000000.0 < p
}

func clip01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func sortGenomes(pop []dnaGenome) {
	for i := 0; i < len(pop)-1; i++ {
		for j := 0; j < len(pop)-1-i; j++ {
			if pop[j].fit > pop[j+1].fit {
				pop[j], pop[j+1] = pop[j+1], pop[j]
			}
		}
	}
}

func cloneGenome(g dnaGenome) dnaGenome {
	return dnaGenome{
		seeds:   append([]uint64(nil), g.seeds...),
		fit:     g.fit,
		acc:     g.acc,
		cluster: g.cluster,
	}
}
