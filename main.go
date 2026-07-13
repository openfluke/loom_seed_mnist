package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/openfluke/chaosglue/loom_seed_mnist/seedmnist"
)

func main() {
	root := "."
	modeFlag := ""
	contFlag := false
	for _, a := range os.Args[1:] {
		switch {
		case a == "--warmth" || a == "-w":
			modeFlag = "warmth"
		case a == "--dna" || a == "-d":
			modeFlag = "dna"
		case a == "--dna-layer" || a == "-3" || a == "dna-layer":
			modeFlag = "dna-layer"
		case a == "--continue" || a == "-c" || a == "continue":
			contFlag = true
		case strings.HasPrefix(a, "--mode="):
			modeFlag = strings.TrimPrefix(a, "--mode=")
		case !strings.HasPrefix(a, "-"):
			switch a {
			case "continue":
				contFlag = true
			case "dna-layer":
				modeFlag = "dna-layer"
			default:
				root = a
			}
		}
	}

	seedsExist := false
	if _, err := os.Stat("mnist.seeds.json"); err == nil {
		seedsExist = true
	}

	needModeAsk := false
	if seedsExist && !contFlag && os.Getenv("LOOM_SEED_MNIST_CONTINUE") == "" && modeFlag == "" {
		action := pickExistingAction()
		switch action {
		case "reload":
			if !seedmnist.RunAll(root) {
				os.Exit(1)
			}
			return
		case "continue":
			seedmnist.SetContinue(true)
			needModeAsk = true
		case "fresh":
			_ = os.Remove("mnist.seeds.json")
			_ = os.Remove("mnist.pop.json")
			seedmnist.SetContinue(false)
			needModeAsk = true
		}
	} else if contFlag {
		seedmnist.SetContinue(true)
		if modeFlag == "" {
			needModeAsk = true
		}
	} else if !seedsExist && modeFlag == "" {
		needModeAsk = true
	}

	mode := parseModeFlag(modeFlag)
	if needModeAsk && modeFlag == "" {
		mode = askTrainMode()
	} else if modeFlag == "" && os.Getenv("LOOM_SEED_MNIST_MODE") != "" {
		mode = seedmnist.ResolveTrainMode()
	}

	seedmnist.SetTrainMode(mode)
	fmt.Printf("selected mode: %s  continue=%v\n", mode, seedmnist.WantContinue())

	if !seedmnist.RunAll(root) {
		os.Exit(1)
	}
}

func pickExistingAction() string {
	fmt.Println()
	fmt.Println("mnist.seeds.json found:")
	fmt.Println("  [1] reload only — He-init from saved seeds (no search)")
	fmt.Println("  [2] continue    — resume search from saved seeds + clusters")
	fmt.Println("  [3] fresh       — delete seeds/pop and pick a train mode")
	fmt.Print("Choice [1]: ")
	reader := bufio.NewReader(os.Stdin)
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	switch line {
	case "2", "continue", "c":
		return "continue"
	case "3", "fresh", "f", "delete":
		return "fresh"
	default:
		return "reload"
	}
}

func parseModeFlag(flag string) seedmnist.TrainMode {
	switch strings.ToLower(flag) {
	case "dna-layer", "dnalayer", "layer", "coord":
		return seedmnist.ModeDNALayer
	case "dna", "neat", "pop", "cluster":
		return seedmnist.ModeDNA
	case "warmth", "warm", "hill", "bits":
		return seedmnist.ModeWarmth
	default:
		return seedmnist.ModeWarmth
	}
}

func askTrainMode() seedmnist.TrainMode {
	fmt.Println()
	fmt.Println("[1] warmth    — single genome · warm-bit hill-climb")
	fmt.Println("[2] dna       — clustered DNA · all layer seeds at once")
	fmt.Println("[3] dna-layer — clustered DNA · one layer seed at a time")
	fmt.Print("Choice [1]: ")
	reader := bufio.NewReader(os.Stdin)
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)
	switch line {
	case "2", "dna", "DNA":
		return seedmnist.ModeDNA
	case "3", "dna-layer", "layer":
		return seedmnist.ModeDNALayer
	default:
		return seedmnist.ModeWarmth
	}
}
