package seedmnist

import (
	"fmt"
	"math"
	"strings"

	"github.com/openfluke/loom/poly"
)

// softFitness is continuous "how warm": lower is better.
// Combines softmax cross-entropy with a margin term so the dial moves
// even when hard accuracy is stuck at ~10%.
func softFitness(net *poly.VolumetricNetwork, samples []Sample) float64 {
	if len(samples) == 0 {
		return 0
	}
	var sum float64
	for _, s := range samples {
		t := poly.NewTensorFromSlice(s.Pixels, 1, len(s.Pixels))
		out, _, _ := poly.ForwardPolymorphic(net, t)
		if out == nil || len(out.Data) == 0 {
			sum += 10
			continue
		}
		sum += softmaxNLL(out.Data, s.Label)
		sum += 0.1 * marginPenalty(out.Data, s.Label)
	}
	return sum / float64(len(samples))
}

func softmaxNLL(logits []float32, label int) float64 {
	maxV := float64(logits[0])
	for i := 1; i < len(logits); i++ {
		if v := float64(logits[i]); v > maxV {
			maxV = v
		}
	}
	var sumExp float64
	for _, x := range logits {
		sumExp += math.Exp(float64(x) - maxV)
	}
	logZ := maxV + math.Log(sumExp+1e-12)
	if label < 0 || label >= len(logits) {
		return logZ
	}
	return logZ - float64(logits[label])
}

func marginPenalty(logits []float32, label int) float64 {
	if label < 0 || label >= len(logits) {
		return 1
	}
	correct := float64(logits[label])
	bestWrong := math.Inf(-1)
	for i, x := range logits {
		if i == label {
			continue
		}
		if v := float64(x); v > bestWrong {
			bestWrong = v
		}
	}
	margin := correct - bestWrong
	if margin >= 1 {
		return 0
	}
	return 1 - margin
}

// probeWarmBits flips each bit once and measures ΔsoftFitness.
// Returns signed warmth per bit (positive = flipping that bit helped).
func probeWarmBits(
	net *poly.VolumetricNetwork,
	layerIdx, in int,
	seed uint64,
	samples []Sample,
	baseFit float64,
) (warmth [64]float64, warmCount int) {
	for bit := 0; bit < 64; bit++ {
		trial := seed ^ (uint64(1) << bit)
		applyLayerSeed(net, layerIdx, trial, in)
		fit := softFitness(net, samples)
		delta := fit - baseFit
		warmth[bit] = -delta
		if warmth[bit] > 1e-6 {
			warmCount++
		}
	}
	applyLayerSeed(net, layerIdx, seed, in)
	return warmth, warmCount
}

func pickWarmBiasedMutation(seed uint64, warmth [64]float64, noise uint64) uint64 {
	switch noise % 5 {
	case 0, 1:
		bit := pickWarmBit(warmth, noise)
		return seed ^ (uint64(1) << bit)
	case 2:
		b0 := pickWarmBit(warmth, noise)
		b1 := pickWarmBit(warmth, noise>>8)
		if b1 == b0 {
			b1 = (b0 + 1 + int((noise>>16)%63)) % 64
		}
		return seed ^ (uint64(1) << b0) ^ (uint64(1) << b1)
	case 3:
		step := (noise >> 6) % 1024
		if step == 0 {
			step = 1
		}
		if noise&1 == 0 {
			return seed + step
		}
		return seed - step
	default:
		return seed ^ noise
	}
}

func pickWarmBit(warmth [64]float64, noise uint64) int {
	var sum float64
	for _, w := range warmth {
		if w > 0 {
			sum += w
		}
	}
	if sum <= 0 {
		return int((noise / 3) % 64)
	}
	target := (float64(noise%1000000) / 1000000.0) * sum
	var acc float64
	for i, w := range warmth {
		if w <= 0 {
			continue
		}
		acc += w
		if acc >= target {
			return i
		}
	}
	return 63
}

func antitheticStep(seed, noise uint64) (plus, minus uint64) {
	step := (noise >> 3) % 4096
	if step == 0 {
		step = 1
	}
	return seed + step, seed - step
}

func thermometer(startFit, bestFit, epochFit float64) string {
	if startFit <= 0 {
		startFit = 1
	}
	improved := (startFit - bestFit) / startFit
	if improved < 0 {
		improved = 0
	}
	if improved > 1 {
		improved = 1
	}
	filled := int(improved * 20)
	bar := strings.Repeat("█", filled) + strings.Repeat("░", 20-filled)

	trend := "·"
	if epochFit < bestFit-1e-4 {
		trend = "▲ warmer"
	} else if epochFit > bestFit+1e-4 {
		trend = "▼ colder than best"
	} else {
		trend = "● at best"
	}
	return fmt.Sprintf("[%s] best↓%.1f%%  %s", bar, improved*100, trend)
}

func printWarmBits(layer int, warmth [64]float64, warmCount int) {
	fmt.Printf("    layer %d warm bits (%d/64): ", layer, warmCount)
	type pair struct {
		bit int
		w   float64
	}
	top := make([]pair, 0, 64)
	for i, w := range warmth {
		if w > 1e-6 {
			top = append(top, pair{i, w})
		}
	}
	for i := 0; i < len(top); i++ {
		for j := i + 1; j < len(top); j++ {
			if top[j].w > top[i].w {
				top[i], top[j] = top[j], top[i]
			}
		}
	}
	limit := 8
	if len(top) < limit {
		limit = len(top)
	}
	if limit == 0 {
		fmt.Println("(all cold — random explore)")
		return
	}
	for i := 0; i < limit; i++ {
		fmt.Printf("b%d(%.3f) ", top[i].bit, top[i].w)
	}
	fmt.Println()
}
