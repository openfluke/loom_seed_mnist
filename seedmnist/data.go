package seedmnist

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// Sample is one MNIST example: flattened pixels + digit label.
type Sample struct {
	Pixels []float32
	Label  int
}

// Dataset holds the full combined MNIST set after shuffle + split.
type Dataset struct {
	Train []Sample
	Val   []Sample
}

// LoadAll downloads (if needed), merges train+test, shuffles, splits 80/20.
func LoadAll(dataDir string, trainFrac float64) (*Dataset, error) {
	if trainFrac <= 0 || trainFrac >= 1 {
		trainFrac = 0.8
	}
	fmt.Println("── MNIST data ──")
	if err := EnsureMNIST(dataDir); err != nil {
		return nil, err
	}

	trainX, err := loadImages(filepath.Join(dataDir, "train-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	trainY, err := loadLabels(filepath.Join(dataDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	testX, err := loadImages(filepath.Join(dataDir, "t10k-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	testY, err := loadLabels(filepath.Join(dataDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	if len(trainX) != len(trainY) || len(testX) != len(testY) {
		return nil, fmt.Errorf("mnist length mismatch")
	}

	all := make([]Sample, 0, len(trainX)+len(testX))
	for i := range trainX {
		all = append(all, Sample{Pixels: trainX[i], Label: trainY[i]})
	}
	for i := range testX {
		all = append(all, Sample{Pixels: testX[i], Label: testY[i]})
	}

	rng := poly.NewSeedRNG(poly.SeedFrom("loom-seed-mnist-shuffle", uint64(len(all))))
	shuffleSamples(all, rng)

	nTrain := int(float64(len(all)) * trainFrac)
	if nTrain < 1 {
		nTrain = 1
	}
	if nTrain >= len(all) {
		nTrain = len(all) - 1
	}
	ds := &Dataset{
		Train: all[:nTrain],
		Val:   all[nTrain:],
	}
	fmt.Printf("  combined %d images (original 60k train + 10k test)\n", len(all))
	fmt.Printf("  split %.0f/%.0f → %d train · %d val\n",
		trainFrac*100, (1-trainFrac)*100, len(ds.Train), len(ds.Val))
	return ds, nil
}

func shuffleSamples(s []Sample, rng *poly.SeedRNG) {
	for i := len(s) - 1; i > 0; i-- {
		j := int(rng.Uint64() % uint64(i+1))
		s[i], s[j] = s[j], s[i]
	}
}

func oneHot(label int) []float32 {
	v := make([]float32, mnistClasses)
	if label >= 0 && label < mnistClasses {
		v[label] = 1
	}
	return v
}

func samplesToTensors(samples []Sample) (inputs []*poly.Tensor[float32], expected []float64) {
	inputs = make([]*poly.Tensor[float32], len(samples))
	expected = make([]float64, len(samples))
	for i, s := range samples {
		inputs[i] = poly.NewTensorFromSlice(s.Pixels, 1, len(s.Pixels))
		expected[i] = float64(s.Label)
	}
	return inputs, expected
}

func sampleSubset(src []Sample, n int, rng *poly.SeedRNG) []Sample {
	if n >= len(src) {
		out := make([]Sample, len(src))
		copy(out, src)
		return out
	}
	// reservoir-ish: shuffle indices via RNG into first n unique picks
	idx := make([]int, len(src))
	for i := range idx {
		idx[i] = i
	}
	for i := len(idx) - 1; i > 0; i-- {
		j := int(rng.Uint64() % uint64(i+1))
		idx[i], idx[j] = idx[j], idx[i]
	}
	out := make([]Sample, n)
	for i := 0; i < n; i++ {
		out[i] = src[idx[i]]
	}
	return out
}
