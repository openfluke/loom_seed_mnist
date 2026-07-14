// Package seedmnist — download MNIST, 80/20 split, layer_seed hill-climb, DeviationMetrics heatmaps.
package seedmnist

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

const (
	runName   = "mnist-seed-classifier"
	runFormat = "chaosglue-seed-mnist-v1"
	seedFile  = "mnist.seeds.json"
)

// SeedFile stores trained layer_seed values only.
type SeedFile struct {
	Format       string      `json:"format"`
	Name         string      `json:"name"`
	Method       string      `json:"method"`
	Dataset      string      `json:"dataset"`
	TrainSamples int         `json:"train_samples"`
	ValSamples   int         `json:"val_samples"`
	TopologySeed uint64      `json:"topology_seed"`
	Sizes        []int       `json:"sizes"`
	Layers       []SeedLayer `json:"layers"`
	TrainAcc     float64     `json:"train_acc"`
	ValAcc       float64     `json:"val_acc"`
	TrainMSE     float32     `json:"train_mse"`
	ValMSE       float32     `json:"val_mse"`
}

// SeedLayer is one dense layer_seed.
type SeedLayer struct {
	Index     int    `json:"index"`
	In        int    `json:"in"`
	Out       int    `json:"out"`
	LayerSeed uint64 `json:"layer_seed"`
	DType     string `json:"dtype"`
}

// RunAll trains, continues, or reloads depending on env/flags and files present.
func RunAll(root string) bool {
	if root == "" {
		root = "."
	}
	dataDir := filepath.Join(root, "data")
	path := filepath.Join(root, seedFile)

	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Loom seed MNIST — layer_seed train · real digits        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")

	_, err := os.Stat(path)
	exists := err == nil
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		fmt.Printf("  FAIL stat %s: %v\n", path, err)
		return false
	}

	if exists && WantContinue() {
		fmt.Println("  mode: CONTINUE search from saved seeds / pop checkpoint")
		return runTrain(root, path, dataDir, true)
	}
	if exists {
		return runReload(path, dataDir)
	}
	return runTrain(root, path, dataDir, false)
}

func runReload(path, dataDir string) bool {
	fmt.Printf("\n── Rerun: %s — trained seeds → He-init (no train) ──\n", seedFile)
	fmt.Println("   continue search: LOOM_SEED_MNIST_CONTINUE=1 ./run.sh dna")
	fmt.Println("   or: ./run.sh continue")
	fmt.Println("   delete file to train from scratch")

	file, err := loadSeedFile(path)
	if err != nil {
		fmt.Printf("  FAIL load: %v\n", err)
		return false
	}
	printSeedSummary(file, path)

	ds, err := LoadAll(dataDir, 0.8)
	if err != nil {
		fmt.Printf("  FAIL dataset: %v\n", err)
		return false
	}

	net, err := buildNetFromFile(file)
	if err != nil {
		fmt.Printf("  FAIL seeds→net: %v\n", err)
		return false
	}

	trainAcc := evalAccuracy(net, ds.Train)
	valAcc := evalAccuracy(net, ds.Val)
	fmt.Printf("\n  reloaded metrics: train acc=%.2f%%  val acc=%.2f%%\n", trainAcc*100, valAcc*100)
	if math.Abs(trainAcc-file.TrainAcc) > 1e-3 || math.Abs(valAcc-file.ValAcc) > 1e-3 {
		fmt.Println("✗ FAIL: accuracy differs from saved baseline")
		return false
	}

	printDeviationHeatmap(net, ds.Val, min(2000, len(ds.Val)), "val (reloaded)")
	printConfusion(net, ds.Val, 12)
	fmt.Println("\n✓ Reload complete — weights from layer_seed He-init only.")
	return true
}

func runTrain(root, path, dataDir string, cont bool) bool {
	if cont {
		fmt.Printf("\n── CONTINUE: resume DNA/warmth search using %s ──\n", seedFile)
	} else {
		fmt.Printf("\n── First run: no %s ──\n", seedFile)
	}
	mode := resolveTrainMode()
	fmt.Printf("  train mode: %s\n", mode)

	ds, err := LoadAll(dataDir, 0.8)
	if err != nil {
		fmt.Printf("  FAIL dataset: %v\n", err)
		return false
	}

	sizes := []int{mnistPixels, 128, 64, mnistClasses}
	dtypes := []string{"float32", "float32", "float32"}
	topo := poly.DenseTopologySeed(runName, sizes)
	fmt.Printf("\nnetwork sizes=%v topology_seed=0x%x\n", sizes, topo)

	manifest, err := poly.BuildDenseManifest(topo, sizes, dtypes)
	if err != nil {
		fmt.Printf("  FAIL BuildDenseManifest: %v\n", err)
		return false
	}

	initSeeds := make([]uint64, len(manifest.Layers))
	for i, l := range manifest.Layers {
		initSeeds[i] = l.LayerSeed
		fmt.Printf("  layer %d %dx%d init_seed=0x%x\n", i, l.In, l.Out, l.LayerSeed)
	}

	var resumeSeeds []uint64
	prevVal := 0.0
	if cont {
		file, err := loadSeedFile(path)
		if err != nil {
			fmt.Printf("  FAIL load resume seeds: %v\n", err)
			return false
		}
		resumeSeeds = make([]uint64, len(file.Layers))
		for i, l := range file.Layers {
			resumeSeeds[i] = l.LayerSeed
			fmt.Printf("  resume layer %d seed=0x%x\n", i, l.LayerSeed)
		}
		prevVal = file.ValAcc
		fmt.Printf("  previous saved val acc=%.2f%%\n", prevVal*100)
	}

	var (
		seeds []uint64
		net   *poly.VolumetricNetwork
	)
	method := string(mode)

	switch mode {
	case ModeDNA:
		res, err := trainDNAPopulation(root, topo, sizes, dtypes, initSeeds, ds.Train, ds.Val, defaultDNAPopConfig(), resumeSeeds)
		if err != nil {
			fmt.Printf("  FAIL dna pop: %v\n", err)
			return false
		}
		seeds, net = res.Seeds, res.Net
		method = "dna-clusters-v2"
	case ModeDNALayer:
		res, err := trainDNALayerWise(root, topo, sizes, dtypes, initSeeds, ds.Train, ds.Val, resumeSeeds)
		if err != nil {
			fmt.Printf("  FAIL dna-layer: %v\n", err)
			return false
		}
		seeds, net = res.Seeds, res.Net
		method = "dna-layer-coord"
	case ModeCascade:
		res, err := trainDNACascade(root, topo, sizes, dtypes, initSeeds, ds.Train, ds.Val, resumeSeeds)
		if err != nil {
			fmt.Printf("  FAIL dna-cascade: %v\n", err)
			return false
		}
		seeds, net = res.Seeds, res.Net
		method = "dna-cascade-v1"
	case ModeMicroFountain:
		res, err := trainMicroFountain(root, topo, sizes, dtypes, initSeeds, ds.Train, ds.Val, resumeSeeds)
		if err != nil {
			fmt.Printf("  FAIL micro-fountain: %v\n", err)
			return false
		}
		seeds, net = res.Seeds, res.Net
		method = "micro-fountain-v1"
	default:
		net, err = poly.BuildDenseVolumetricFromManifest(manifest)
		if err != nil {
			fmt.Printf("  FAIL build: %v\n", err)
			return false
		}
		last := net.GetLayer(0, 0, 0, len(manifest.Layers)-1)
		last.Activation = poly.ActivationLinear
		if len(resumeSeeds) == len(initSeeds) {
			seeds = append([]uint64(nil), resumeSeeds...)
			for i, s := range seeds {
				applyLayerSeed(net, i, s, sizes[i])
			}
		} else {
			seeds = append([]uint64(nil), initSeeds...)
		}
		trainLayerSeeds(net, seeds, sizes, ds.Train, ds.Val, defaultTrainConfig())
		method = "warmth-bits"
	}

	trainAcc := evalAccuracy(net, ds.Train)
	valAcc := evalAccuracy(net, ds.Val)
	trainMSE := evalMSE(net, sampleSubset(ds.Train, min(4000, len(ds.Train)), poly.NewSeedRNG(1)))
	valMSE := evalMSE(net, sampleSubset(ds.Val, min(2000, len(ds.Val)), poly.NewSeedRNG(2)))

	fmt.Printf("\n── AFTER seed training (%s) ──\n", method)
	fmt.Printf("  train acc=%.2f%%  val acc=%.2f%%\n", trainAcc*100, valAcc*100)
	if cont {
		fmt.Printf("  vs previous saved val %.2f%% → %.2f%%\n", prevVal*100, valAcc*100)
	}
	fmt.Printf("  train MSE≈%.5f  val MSE≈%.5f (subsampled)\n", trainMSE, valMSE)

	printDeviationHeatmap(net, ds.Val, min(2000, len(ds.Val)), "val AFTER")
	printDeviationHeatmap(net, ds.Train, min(2000, len(ds.Train)), "train AFTER")

	baselineSeeds := initSeeds
	if len(resumeSeeds) == len(initSeeds) {
		// compare against topology init still, for the showcase table
		baselineSeeds = initSeeds
	}
	initNet, err := rebuildNet(topo, sizes, dtypes, baselineSeeds)
	if err != nil {
		fmt.Printf("  FAIL rebuild init: %v\n", err)
		return false
	}
	printBeforeAfter(initNet, net, ds.Val, min(2000, len(ds.Val)), "validation")
	printBeforeAfter(initNet, net, ds.Train, min(2000, len(ds.Train)), "train")

	if valAcc <= evalAccuracy(initNet, ds.Val) {
		fmt.Println("\n✗ FAIL: trained val accuracy did not beat topology init")
		return false
	}

	fmt.Println("\n── Trained seeds (hall-of-fame) ──")
	for i, seed := range seeds {
		changed := ""
		if seed != initSeeds[i] {
			changed = " *"
		}
		fmt.Printf("  layer %d trained_seed=0x%x%s\n", i, seed, changed)
	}
	printConfusion(net, ds.Val, 15)

	layers := make([]SeedLayer, len(seeds))
	for i, l := range manifest.Layers {
		layers[i] = SeedLayer{
			Index: i, In: l.In, Out: l.Out,
			LayerSeed: seeds[i], DType: l.DType,
		}
	}
	file := SeedFile{
		Format:       runFormat,
		Name:         runName,
		Method:       method,
		Dataset:      "MNIST (all 70k, 80/20 shuffle split)",
		TrainSamples: len(ds.Train),
		ValSamples:   len(ds.Val),
		TopologySeed: topo,
		Sizes:        sizes,
		Layers:       layers,
		TrainAcc:     trainAcc,
		ValAcc:       valAcc,
		TrainMSE:     trainMSE,
		ValMSE:       valMSE,
	}
	if err := saveSeedFile(path, file); err != nil {
		fmt.Printf("  FAIL save: %v\n", err)
		return false
	}
	st, _ := os.Stat(path)
	fmt.Printf("\n✓ Saved seeds-only %s (%d bytes) method=%s\n", path, st.Size(), method)
	fmt.Println("  reload: ./run.sh")
	fmt.Println("  continue search: ./run.sh continue   (keeps mnist.pop.json clusters)")
	return true
}

func rebuildNet(topo uint64, sizes []int, dtypes []string, seeds []uint64) (*poly.VolumetricNetwork, error) {
	manifest, err := poly.BuildDenseManifest(topo, sizes, dtypes)
	if err != nil {
		return nil, err
	}
	for i, s := range seeds {
		manifest.Layers[i].LayerSeed = s
	}
	net, err := poly.BuildDenseVolumetricFromManifest(manifest)
	if err != nil {
		return nil, err
	}
	last := net.GetLayer(0, 0, 0, len(manifest.Layers)-1)
	last.Activation = poly.ActivationLinear
	return net, nil
}

func buildNetFromFile(f SeedFile) (*poly.VolumetricNetwork, error) {
	dtypes := make([]string, len(f.Layers))
	seeds := make([]uint64, len(f.Layers))
	for i, l := range f.Layers {
		dtypes[i] = l.DType
		seeds[i] = l.LayerSeed
	}
	return rebuildNet(f.TopologySeed, f.Sizes, dtypes, seeds)
}

func printSeedSummary(f SeedFile, path string) {
	st, _ := os.Stat(path)
	size := int64(0)
	if st != nil {
		size = st.Size()
	}
	fmt.Printf("\n  file: %s (%d bytes)\n", path, size)
	fmt.Printf("  dataset: %s (%d train / %d val)\n", f.Dataset, f.TrainSamples, f.ValSamples)
	fmt.Printf("  topology_seed=0x%x sizes=%v\n", f.TopologySeed, f.Sizes)
	for _, l := range f.Layers {
		fmt.Printf("    layer %d %dx%d %s seed=0x%x\n", l.Index, l.In, l.Out, l.DType, l.LayerSeed)
	}
	if f.Method != "" {
		fmt.Printf("  method: %s\n", f.Method)
	}
	fmt.Printf("  saved acc: train=%.2f%% val=%.2f%%\n", f.TrainAcc*100, f.ValAcc*100)
}

func saveSeedFile(path string, f SeedFile) error {
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadSeedFile(path string) (SeedFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SeedFile{}, err
	}
	var f SeedFile
	if err := json.Unmarshal(data, &f); err != nil {
		return SeedFile{}, err
	}
	if f.Format != runFormat {
		return SeedFile{}, fmt.Errorf("unknown format %q — delete %s and rerun", f.Format, seedFile)
	}
	return f, nil
}
