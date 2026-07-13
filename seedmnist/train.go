package seedmnist

import (
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/openfluke/loom/poly"
)

// TrainConfig controls seed hill-climb budget.
type TrainConfig struct {
	Epochs       int
	MutPerLayer  int
	FitnessBatch int
	HeatmapVal   int
	HeatmapEvery int // epochs between deviation heatmaps
	FullEvalEvery int
}

func defaultTrainConfig() TrainConfig {
	cfg := TrainConfig{
		Epochs:        40,
		MutPerLayer:   80,
		FitnessBatch:  512,
		HeatmapVal:    2000,
		HeatmapEvery:  5,
		FullEvalEvery: 5,
	}
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		cfg.Epochs = 12
		cfg.MutPerLayer = 40
		cfg.FitnessBatch = 256
		cfg.HeatmapVal = 500
		cfg.HeatmapEvery = 3
		cfg.FullEvalEvery = 4
	}
	if v := envInt("LOOM_SEED_MNIST_EPOCHS"); v > 0 {
		cfg.Epochs = v
	}
	if v := envInt("LOOM_SEED_MNIST_MUT"); v > 0 {
		cfg.MutPerLayer = v
	}
	return cfg
}

func envInt(key string) int {
	s := os.Getenv(key)
	if s == "" {
		return 0
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return n
}

func trainLayerSeeds(
	net *poly.VolumetricNetwork,
	seeds []uint64,
	sizes []int,
	train []Sample,
	val []Sample,
	cfg TrainConfig,
) {
	rng := poly.NewSeedRNG(poly.SeedFrom("loom-seed-mnist-train", seeds[0]))
	fitness := sampleSubset(train, cfg.FitnessBatch, rng)

	startSoft := softFitness(net, fitness)
	bestSoft := startSoft
	bestMSE := evalMSE(net, fitness)
	bestAcc := evalAccuracy(net, fitness)

	fmt.Printf("\n── Seed train (warmth-guided): %d epochs × %d mut/layer · batch=%d ──\n",
		cfg.Epochs, cfg.MutPerLayer, cfg.FitnessBatch)
	fmt.Println("  soft fitness = softmax-NLL + margin (continuous warmer/colder dial)")
	fmt.Println("  each epoch: probe 64 bit flips → warm-bit map → biased mutations")
	fmt.Printf("  start: soft=%.4f  mse=%.4f  acc=%.1f%%\n", startSoft, bestMSE, bestAcc*100)

	printDeviationHeatmap(net, val, min(cfg.HeatmapVal, len(val)), "val BEFORE (epoch 0)")

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		fitness = sampleSubset(train, cfg.FitnessBatch, rng)
		// Re-score current seeds on new batch so comparisons stay consistent
		for li := range seeds {
			applyLayerSeed(net, li, seeds[li], sizes[li])
		}
		curSoft := softFitness(net, fitness)
		if curSoft < bestSoft {
			bestSoft = curSoft
		}

		accepts := 0
		warmer := 0
		colder := 0
		var layerWarm [32][64]float64
		var layerWarmN [32]int

		for li := range seeds {
			in := sizes[li]
			base := softFitness(net, fitness)
			warmth, wn := probeWarmBits(net, li, in, seeds[li], fitness, base)
			layerWarm[li] = warmth
			layerWarmN[li] = wn

			for m := 0; m < cfg.MutPerLayer; m++ {
				noise := rng.Uint64()
				var trial uint64
				if m%4 == 0 {
					// antithetic: try +δ and −δ, keep the warmer one if any
					plus, minus := antitheticStep(seeds[li], noise)
					applyLayerSeed(net, li, plus, in)
					fp := softFitness(net, fitness)
					applyLayerSeed(net, li, minus, in)
					fm := softFitness(net, fitness)
					trial, curSoft = plus, fp
					if fm < fp {
						trial, curSoft = minus, fm
					}
				} else {
					trial = pickWarmBiasedMutation(seeds[li], warmth, noise)
					applyLayerSeed(net, li, trial, in)
					curSoft = softFitness(net, fitness)
				}

				if curSoft < bestSoft {
					bestSoft = curSoft
					seeds[li] = trial
					accepts++
					warmer++
				} else if curSoft < base {
					// locally warmer than layer base — still climb even if not global best
					seeds[li] = trial
					base = curSoft
					accepts++
					warmer++
				} else {
					colder++
				}
			}
			applyLayerSeed(net, li, seeds[li], in)
		}

		bestMSE = evalMSE(net, fitness)
		bestAcc = evalAccuracy(net, fitness)
		epochSoft := softFitness(net, fitness)
		fmt.Printf("  epoch %02d/%d  soft=%.4f  mse=%.4f  acc=%.1f%%  accepts=%d  warm/cold=%d/%d\n",
			epoch, cfg.Epochs, bestSoft, bestMSE, bestAcc*100, accepts, warmer, colder)
		fmt.Printf("           %s\n", thermometer(startSoft, bestSoft, epochSoft))
		for li := range seeds {
			printWarmBits(li, layerWarm[li], layerWarmN[li])
		}

		if cfg.HeatmapEvery > 0 && epoch%cfg.HeatmapEvery == 0 {
			printDeviationHeatmap(net, val, min(cfg.HeatmapVal, len(val)),
				fmt.Sprintf("val epoch %d", epoch))
		}
		if cfg.FullEvalEvery > 0 && epoch%cfg.FullEvalEvery == 0 {
			trainAcc := evalAccuracy(net, train)
			valAcc := evalAccuracy(net, val)
			fmt.Printf("  ▸ full eval  train acc=%.2f%%  val acc=%.2f%%\n",
				trainAcc*100, valAcc*100)
		}
	}
}

func applyLayerSeed(net *poly.VolumetricNetwork, layerIdx int, seed uint64, in int) {
	l := net.GetLayer(0, 0, 0, layerIdx)
	poly.InitWeightStoreHeSeeded(l.WeightStore, in, seed)
}

func mutateSeed(seed, noise uint64) uint64 {
	switch noise % 3 {
	case 0:
		bit := (noise / 3) % 64
		return seed ^ (1 << bit)
	case 1:
		return seed + (noise >> 6) + 1
	default:
		return seed ^ noise
	}
}

func evalMSE(net *poly.VolumetricNetwork, samples []Sample) float32 {
	var sum float32
	for _, s := range samples {
		t := poly.NewTensorFromSlice(s.Pixels, 1, len(s.Pixels))
		out, _, _ := poly.ForwardPolymorphic(net, t)
		if out == nil {
			continue
		}
		target := oneHot(s.Label)
		for j := range out.Data {
			d := out.Data[j] - target[j]
			sum += d * d
		}
	}
	return sum / float32(len(samples))
}

func evalAccuracy(net *poly.VolumetricNetwork, samples []Sample) float64 {
	if len(samples) == 0 {
		return 0
	}
	correct := 0
	for _, s := range samples {
		if predictClass(net, s.Pixels) == s.Label {
			correct++
		}
	}
	return float64(correct) / float64(len(samples))
}

func predictClass(net *poly.VolumetricNetwork, pixels []float32) int {
	t := poly.NewTensorFromSlice(pixels, 1, len(pixels))
	out, _, _ := poly.ForwardPolymorphic(net, t)
	if out == nil || len(out.Data) == 0 {
		return -1
	}
	best := 0
	bestV := out.Data[0]
	for i := 1; i < len(out.Data); i++ {
		if out.Data[i] > bestV {
			bestV = out.Data[i]
			best = i
		}
	}
	return best
}

func printDeviationHeatmap(net *poly.VolumetricNetwork, samples []Sample, n int, title string) {
	if n > len(samples) {
		n = len(samples)
	}
	subset := samples[:n]
	inputs, expected := samplesToTensors(subset)
	m, err := poly.EvaluateNetworkPolymorphic(net, inputs, expected)
	if err != nil {
		fmt.Printf("  FAIL evaluation %s: %v\n", title, err)
		return
	}
	fmt.Printf("\n── %s · n=%d (DeviationMetrics 0–100) ──\n", title, n)
	m.PrintSummary()
	printSampleScoreStrip(m)
	acc := evalAccuracy(net, subset)
	fmt.Printf("  classification accuracy on this slice: %.2f%%\n", acc*100)
}

func printSampleScoreStrip(m *poly.DeviationMetrics) {
	if m == nil || len(m.Results) == 0 {
		return
	}
	const maxShow = 120
	fmt.Printf("  per-sample quality: ")
	limit := len(m.Results)
	if limit > maxShow {
		limit = maxShow
	}
	for i := 0; i < limit; i++ {
		q := math.Max(0, 100-m.Results[i].Deviation)
		fmt.Printf("%c", sampleQualityChar(q))
	}
	if len(m.Results) > maxShow {
		fmt.Printf("…+%d", len(m.Results)-maxShow)
	}
	fmt.Printf("\n  legend: ░ poor · ▒ fair · ▓ good · █ excellent (exact class → █)\n")
}

func sampleQualityChar(score float64) rune {
	switch {
	case score >= 90:
		return '█'
	case score >= 70:
		return '▓'
	case score >= 50:
		return '▒'
	default:
		return '░'
	}
}

func printBeforeAfter(initNet, trainedNet *poly.VolumetricNetwork, samples []Sample, n int, split string) {
	if n > len(samples) {
		n = len(samples)
	}
	subset := samples[:n]
	inputs, expected := samplesToTensors(subset)
	results, err := poly.MultiNetworkEvaluation(map[string]*poly.VolumetricNetwork{
		"init seeds":    initNet,
		"trained seeds": trainedNet,
	}, inputs, expected)
	if err != nil {
		fmt.Printf("  FAIL before/after: %v\n", err)
		return
	}
	fmt.Printf("\n── %s init vs trained (n=%d) ──\n", split, n)
	poly.PrintMultiNetworkSummary(results)
	fmt.Printf("  init acc=%.2f%%  trained acc=%.2f%%\n",
		evalAccuracy(initNet, subset)*100, evalAccuracy(trainedNet, subset)*100)
}

func printConfusion(net *poly.VolumetricNetwork, samples []Sample, maxRows int) {
	fmt.Println("\n  sample predictions:")
	if maxRows > len(samples) {
		maxRows = len(samples)
	}
	for i := 0; i < maxRows; i++ {
		pred := predictClass(net, samples[i].Pixels)
		mark := " "
		if pred == samples[i].Label {
			mark = "✓"
		} else {
			mark = "✗"
		}
		fmt.Printf("    [%02d] actual=%d pred=%d %s\n", i+1, samples[i].Label, pred, mark)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
