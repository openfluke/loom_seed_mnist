package seedmnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	mnistPixels = 28 * 28
	mnistClasses = 10
)

var mnistFiles = []struct {
	name string
	urls []string
}{
	{
		name: "train-images-idx3-ubyte.gz",
		urls: []string{
			"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
			"https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
		},
	},
	{
		name: "train-labels-idx1-ubyte.gz",
		urls: []string{
			"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
			"https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
		},
	},
	{
		name: "t10k-images-idx3-ubyte.gz",
		urls: []string{
			"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
			"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
		},
	},
	{
		name: "t10k-labels-idx1-ubyte.gz",
		urls: []string{
			"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
			"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
		},
	},
}

// EnsureMNIST downloads MNIST gzip IDX files into dataDir if missing.
func EnsureMNIST(dataDir string) error {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}
	for _, f := range mnistFiles {
		path := filepath.Join(dataDir, f.name)
		if st, err := os.Stat(path); err == nil && st.Size() > 0 {
			fmt.Printf("  cached %s (%d bytes)\n", f.name, st.Size())
			continue
		}
		fmt.Printf("  downloading %s …\n", f.name)
		var last error
		for _, url := range f.urls {
			if err := downloadFile(url, path); err != nil {
				last = err
				fmt.Printf("    mirror fail %s: %v\n", url, err)
				continue
			}
			last = nil
			break
		}
		if last != nil {
			return fmt.Errorf("download %s: %w", f.name, last)
		}
		st, _ := os.Stat(path)
		fmt.Printf("  ✓ %s (%d bytes)\n", f.name, st.Size())
	}
	return nil
}

func downloadFile(url, dest string) error {
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	tmp := dest + ".partial"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, dest)
}

func loadImages(path string) ([][]float32, error) {
	raw, err := readGzip(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 16 {
		return nil, fmt.Errorf("images too short: %s", path)
	}
	magic := binary.BigEndian.Uint32(raw[0:4])
	if magic != 2051 {
		return nil, fmt.Errorf("bad image magic 0x%x in %s", magic, path)
	}
	n := int(binary.BigEndian.Uint32(raw[4:8]))
	rows := int(binary.BigEndian.Uint32(raw[8:12]))
	cols := int(binary.BigEndian.Uint32(raw[12:16]))
	if rows*cols != mnistPixels {
		return nil, fmt.Errorf("unexpected image size %dx%d", rows, cols)
	}
	need := 16 + n*mnistPixels
	if len(raw) < need {
		return nil, fmt.Errorf("truncated images file: have %d need %d", len(raw), need)
	}
	out := make([][]float32, n)
	off := 16
	for i := 0; i < n; i++ {
		pix := make([]float32, mnistPixels)
		for j := 0; j < mnistPixels; j++ {
			pix[j] = float32(raw[off+j]) / 255.0
		}
		out[i] = pix
		off += mnistPixels
	}
	return out, nil
}

func loadLabels(path string) ([]int, error) {
	raw, err := readGzip(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("labels too short: %s", path)
	}
	magic := binary.BigEndian.Uint32(raw[0:4])
	if magic != 2049 {
		return nil, fmt.Errorf("bad label magic 0x%x in %s", magic, path)
	}
	n := int(binary.BigEndian.Uint32(raw[4:8]))
	if len(raw) < 8+n {
		return nil, fmt.Errorf("truncated labels file")
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = int(raw[8+i])
	}
	return out, nil
}

func readGzip(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	return io.ReadAll(zr)
}
