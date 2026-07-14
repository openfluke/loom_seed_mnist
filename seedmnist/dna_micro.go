package seedmnist

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

/*
Micro-fountain (mode 5) — regional seed bursts + LT consolidate + mega:

  For each burst round r = 1…R:
    μ) MICRO  — per digit class c∈{0…9}: short DNA islands scoring soft-fitness
       on class-c samples (plus a little mix). Keep s*_c if full-val HOF rises
       or region improves without tanking full-val.
       Then majority-bit vote across s*_c → seed-side consolidate (still He-init).

    κ) L1 FOUNTAIN — PackNetworkWeights(HeInit(s*_c)) → RecoverWeightBlobs →
       unpack specialists → ensemble ForwardArgmax (leaves pure seed deploy for
       seeds.json; Master is the weight-space combo experiment).

    Μ) stash L1 Master cargo (packed K experts) for mega.

  MEGA — Level-2 RecoverWeightBlobs over padded L1 cargos → rebuild zoo → vote.

  Burst again from best seed HOF so exploration stays seed-manifold; fountain
  recombines the micro improvements into weight space each round.
*/

func printMicroFountainEquation() {
	fmt.Println(`
── Micro-fountain (mode 5) — micro seeds → LT consolidate → mega ──
  μ) per-digit DNA micro-bursts (regional soft-fitness)
  κ) Pack(HeInit(s*_c)) → RecoverWeightBlobs → ensemble (L1)
  Μ) RecoverWeightBlobs over L1 Master cargos → mega zoo (L2)
  hof_seeds = best full-val seed genome (deployable)
  fountain  = experimental weight-space ensemble (may leave seed manifold)`)
}

func trainMicroFountain(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	initSeeds []uint64,
	train, val []Sample,
	resumeFrom []uint64,
) (*dnaTrainResult, error) {
	printMicroFountainEquation()

	bursts := 3
	microGens := 8
	microClusters := 3
	microPer := 4
	lossRate := 0.30
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		bursts = 2
		microGens = 3
		microClusters = 2
		microPer = 3
	}
	if v := envInt("LOOM_SEED_MNIST_BURSTS"); v > 0 {
		bursts = v
	}
	if v := envInt("LOOM_SEED_MNIST_GEN"); v > 0 {
		microGens = maxInt(2, v/4)
	}

	rng := poly.NewSeedRNG(poly.SeedFrom("loom-seed-mnist-micro-fountain", initSeeds[0]))
	seeds := append([]uint64(nil), initSeeds...)
	if len(resumeFrom) == len(initSeeds) {
		seeds = append([]uint64(nil), resumeFrom...)
		fmt.Println("  ▶ micro-fountain resuming from saved seeds")
	}

	hofNet, err := rebuildNet(topo, sizes, dtypes, seeds)
	if err != nil {
		return nil, err
	}
	hofVal := evalAccuracy(hofNet, val)
	fmt.Printf("\n── Micro-fountain · start hof val=%.2f%% · bursts=%d ──\n", hofVal*100, bursts)
	printDeviationHeatmap(hofNet, val, min(2000, len(val)), "val BEFORE (micro-fountain)")

	cfg := defaultDNAPopConfig()
	cfg.Generations = microGens
	cfg.Clusters = microClusters
	cfg.PerCluster = microPer
	cfg.FitnessBatch = min(256, cfg.FitnessBatch)
	if os.Getenv("LOOM_SEED_MNIST_QUICK") == "1" {
		cfg.FitnessBatch = 128
		cfg.HeatmapVal = 500
	}

	globalGen := 0
	var cargos [][]byte
	bestFountainVal := 0.0

	for burst := 1; burst <= bursts; burst++ {
		fmt.Printf("\n╔══ BURST %d/%d · micro → L1 fountain ══╗\n", burst, bursts)

		regional := make([][]uint64, 0, mnistClasses)
		for c := 0; c < mnistClasses; c++ {
			regionFit := mixRegion(train, c, 0.15, 384, rng)
			if len(regionFit) < 32 {
				fmt.Printf("  · digit %d skipped (too few samples)\n", c)
				continue
			}
			fmt.Printf("\n── μ digit %d · fit_n=%d · %d gens · free L0 ──\n",
				c, len(regionFit), cfg.Generations)

			best, regionVal, gens, err := dnaSearchRegion(
				root, topo, sizes, dtypes, seeds, []int{0},
				regionFit, val, cfg, rng, &globalGen,
			)
			if err != nil {
				return nil, err
			}
			candNet, err := rebuildNet(topo, sizes, dtypes, best)
			if err != nil {
				return nil, err
			}
			fullVal := evalAccuracy(candNet, val)
			regAcc := evalAccuracy(candNet, filterLabel(val, c))
			keep := false
			if fullVal > hofVal+1e-6 {
				seeds = best
				hofVal = fullVal
				hofNet = candNet
				keep = true
				fmt.Printf("  ★ digit %d lifted HOF full-val → %.2f%%  (regionVal=%.2f%% regAcc=%.1f%% gens=%d)\n",
					c, hofVal*100, regionVal*100, regAcc*100, gens)
			} else if regAcc > evalAccuracy(hofNet, filterLabel(val, c))+0.01 &&
				fullVal+0.005 >= hofVal {
				// micro improvement on the digit without tanking global
				seeds = best
				hofNet = candNet
				hofVal = fullVal
				keep = true
				fmt.Printf("  ◆ digit %d micro-accept  full=%.2f%% regAcc=%.1f%%\n",
					c, fullVal*100, regAcc*100)
			} else {
				fmt.Printf("  · digit %d held  full=%.2f%% ≤ hof %.2f%%  regAcc=%.1f%%\n",
					c, fullVal*100, hofVal*100, regAcc*100)
			}
			if keep {
				regional = append(regional, append([]uint64(nil), seeds...))
			} else {
				regional = append(regional, append([]uint64(nil), best...))
			}
		}

		// Seed-side consolidate: majority bit-vote across regional hof genomes.
		if len(regional) >= 2 {
			voted := majoritySeedVote(regional)
			vNet, err := rebuildNet(topo, sizes, dtypes, voted)
			if err != nil {
				return nil, err
			}
			vAcc := evalAccuracy(vNet, val)
			fmt.Printf("\n── seed consolidate (majority bit-vote) val=%.2f%% ──\n", vAcc*100)
			if vAcc > hofVal+1e-6 {
				seeds = voted
				hofVal = vAcc
				hofNet = vNet
				fmt.Printf("  ★ seed-vote UPDATED hof val=%.2f%%\n", hofVal*100)
			} else {
				fmt.Printf("  · seed-vote held (%.2f%% ≤ hof %.2f%%)\n", vAcc*100, hofVal*100)
			}
		}

		// κ) L1 fountain over regional HeInit weight blobs
		if len(regional) < 2 {
			fmt.Println("  · L1 fountain skipped (need ≥2 regional genomes)")
			continue
		}
		blobs := make([][]byte, 0, len(regional))
		origLens := make([]int, 0, len(regional))
		for i, rs := range regional {
			net, err := rebuildNet(topo, sizes, dtypes, rs)
			if err != nil {
				return nil, err
			}
			b, err := poly.PackNetworkWeights(net)
			if err != nil {
				return nil, fmt.Errorf("pack region %d: %w", i, err)
			}
			blobs = append(blobs, b)
			origLens = append(origLens, len(b))
		}
		padded := padBlobsEqual(blobs)
		fseed := poly.SeedFrom("loom-seed-mnist-l1-fountain", uint64(burst), uint64(len(padded)), uint64(len(padded[0])))
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
		fmt.Printf("  ensemble val=%.2f%%  (seed hof=%.2f%%)\n", fVal*100, hofVal*100)
		if fVal > bestFountainVal {
			bestFountainVal = fVal
			fmt.Printf("  ★ fountain HOF ensemble val=%.2f%%\n", bestFountainVal*100)
		}
		// If a single recovered expert beats seed hof, keep reporting it (weights off-manifold).
		for i, e := range experts {
			ea := evalAccuracy(e, val)
			if ea > hofVal+0.02 {
				fmt.Printf("  ⚠ expert[%d] val=%.2f%% beats seed hof — off seed-manifold (not saved as seeds)\n",
					i, ea*100)
			}
		}

		cargo, err := packExpertsCargo(experts)
		if err != nil {
			return nil, err
		}
		cargos = append(cargos, cargo)
		fmt.Printf("  stashed L1 cargo %dB (total cargos=%d)\n", len(cargo), len(cargos))

		_ = savePopCheckpoint(filepath.Join(root, popCheckpointFile), PopCheckpoint{
			Format:     popFormat,
			Generation: globalGen,
			Clusters:   cfg.Clusters,
			PerCluster: cfg.PerCluster,
			HofSeeds:   seeds,
			HofValAcc:  hofVal,
			Genomes:    nil,
		})
	}

	// Μ) mega consolidate consolidations
	if len(cargos) >= 2 {
		fmt.Printf("\n╔══ MEGA FOUNTAIN — consolidate the consolidations (%d L1 cargos) ══╗\n", len(cargos))
		padded := padBlobsEqual(cargos)
		mseed := poly.SeedFrom("loom-seed-mnist-mega-fountain", uint64(len(padded)), uint64(len(padded[0])))
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
				// recovered cargo may be padded — trim to packed size if needed
				if err := poly.UnpackNetworkWeights(net, trimPackedOrRaw(eb, net)); err != nil {
					// try unpadded original length from Pack of current hof
					ref, _ := poly.PackNetworkWeights(hofNet)
					if ref != nil && len(eb) >= len(ref) {
						if err2 := poly.UnpackNetworkWeights(net, eb[:len(ref)]); err2 != nil {
							fmt.Printf("  · expert unpack failed: %v\n", err2)
							continue
						}
					} else {
						fmt.Printf("  · expert unpack failed: %v\n", err)
						continue
					}
				}
				megaExperts = append(megaExperts, net)
			}
		}
		if len(megaExperts) > 0 {
			mega := &poly.FountainMaster{Experts: megaExperts, K: len(megaExperts)}
			mVal := evalMasterAcc(mega, sampleSubset(val, min(4000, len(val)), rng))
			fmt.Printf("  mega ensemble val≈%.2f%%  (on %d val)  seed hof=%.2f%%  best L1 fountain=%.2f%%\n",
				mVal*100, min(4000, len(val)), hofVal*100, bestFountainVal*100)
			if mVal > bestFountainVal {
				bestFountainVal = mVal
			}
		}
	} else {
		fmt.Println("\n── MEGA skipped (need ≥2 burst cargos) ──")
	}

	printDeviationHeatmap(hofNet, val, min(cfg.HeatmapVal, len(val)), "val AFTER micro-fountain (seed hof)")
	fmt.Printf("\n  ▸ micro-fountain done  seed_train=%.2f%%  seed_val=%.2f%%  fountain_best=%.2f%%  gens≈%d\n",
		evalAccuracy(hofNet, train)*100, hofVal*100, bestFountainVal*100, globalGen)

	return &dnaTrainResult{
		Seeds:      seeds,
		Net:        hofNet,
		Generation: globalGen,
		Clusters:   cfg.Clusters,
	}, nil
}

// dnaSearchRegion is free-set DNA scored on a regional fitness set; HOF by full val.
func dnaSearchRegion(
	root string,
	topo uint64,
	sizes []int,
	dtypes []string,
	baseSeeds []uint64,
	free []int,
	regionFit, val []Sample,
	cfg DNAPopConfig,
	rng *poly.SeedRNG,
	globalGen *int,
) (bestSeeds []uint64, bestVal float64, gensRun int, err error) {
	// Soft fitness on regional samples; HOF still tracks full val inside free-set search.
	return dnaSearchFreeSet(root, topo, sizes, dtypes, baseSeeds, free, regionFit, val, cfg, rng, globalGen)
}

func filterLabel(samples []Sample, label int) []Sample {
	out := make([]Sample, 0, len(samples)/10)
	for _, s := range samples {
		if s.Label == label {
			out = append(out, s)
		}
	}
	return out
}

// mixRegion: mostly class-label samples + otherFrac random others, capped at n.
func mixRegion(src []Sample, label int, otherFrac float64, n int, rng *poly.SeedRNG) []Sample {
	primary := filterLabel(src, label)
	if len(primary) == 0 {
		return nil
	}
	out := sampleSubset(primary, min(n, len(primary)), rng)
	otherN := int(float64(len(out)) * otherFrac)
	if otherN < 1 {
		return out
	}
	others := make([]Sample, 0, otherN)
	for len(others) < otherN {
		s := src[int(rng.Uint64()%uint64(len(src)))]
		if s.Label != label {
			others = append(others, s)
		}
	}
	out = append(out, others...)
	shuffleSamples(out, rng)
	return out
}

func majoritySeedVote(cands [][]uint64) []uint64 {
	if len(cands) == 0 {
		return nil
	}
	nLayers := len(cands[0])
	out := make([]uint64, nLayers)
	for li := 0; li < nLayers; li++ {
		var votes [64]int
		for _, s := range cands {
			if li >= len(s) {
				continue
			}
			for b := 0; b < 64; b++ {
				if (s[li]>>uint(b))&1 == 1 {
					votes[b]++
				}
			}
		}
		var v uint64
		thresh := (len(cands) + 1) / 2
		for b := 0; b < 64; b++ {
			if votes[b] >= thresh {
				v |= uint64(1) << uint(b)
			}
		}
		out[li] = v
	}
	return out
}

func padBlobsEqual(blobs [][]byte) [][]byte {
	max := 0
	for _, b := range blobs {
		if len(b) > max {
			max = len(b)
		}
	}
	out := make([][]byte, len(blobs))
	for i, b := range blobs {
		p := make([]byte, max)
		copy(p, b)
		out[i] = p
	}
	return out
}

func packExpertsCargo(experts []*poly.VolumetricNetwork) ([]byte, error) {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(experts))); err != nil {
		return nil, err
	}
	for i, e := range experts {
		if e == nil {
			return nil, fmt.Errorf("nil expert %d", i)
		}
		b, err := poly.PackNetworkWeights(e)
		if err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(b))); err != nil {
			return nil, err
		}
		if _, err := buf.Write(b); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func unpackExpertsCargo(blob []byte) ([][]byte, error) {
	r := bytes.NewReader(blob)
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n == 0 || n > 256 {
		return nil, fmt.Errorf("bad expert count %d", n)
	}
	out := make([][]byte, 0, n)
	for i := uint32(0); i < n; i++ {
		var ln uint32
		if err := binary.Read(r, binary.LittleEndian, &ln); err != nil {
			return nil, err
		}
		if ln == 0 || ln > 64<<20 {
			return nil, fmt.Errorf("bad blob len %d", ln)
		}
		b := make([]byte, ln)
		if _, err := r.Read(b); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}

func trimPackedOrRaw(blob []byte, net *poly.VolumetricNetwork) []byte {
	ref, err := poly.PackNetworkWeights(net)
	if err != nil || ref == nil {
		return blob
	}
	if len(blob) >= len(ref) {
		return blob[:len(ref)]
	}
	return blob
}

func evalMasterAcc(m *poly.FountainMaster, samples []Sample) float64 {
	if m == nil || len(samples) == 0 {
		return 0
	}
	ok := 0
	for _, s := range samples {
		in := poly.NewTensorFromSlice(s.Pixels, 1, len(s.Pixels))
		pred, err := m.ForwardArgmax(in)
		if err == nil && pred == s.Label {
			ok++
		}
	}
	return float64(ok) / float64(len(samples))
}
