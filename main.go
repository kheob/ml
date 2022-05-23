package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/kheob/ml/helpers"
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

func CreateNetwork(input, hidden, output int, rate float64) Network {
	net := Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}

	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, helpers.RandomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, helpers.RandomArray(net.hiddens*net.outputs, float64(net.hiddens)))

	return net
}

func (net Network) Predict(inputData []float64) mat.Matrix {
	// forward propogation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := helpers.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := helpers.Apply(helpers.Sigmoid, hiddenInputs)
	finalInputs := helpers.Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := helpers.Apply(helpers.Sigmoid, finalInputs)

	return finalOutputs
}

func (net *Network) Train(inputData []float64, targetData []float64) {
	// forward propogation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := helpers.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := helpers.Apply(helpers.Sigmoid, hiddenInputs)
	finalInputs := helpers.Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := helpers.Apply(helpers.Sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := helpers.Subtract(targets, finalOutputs)
	hiddenErrors := helpers.Dot(net.outputWeights.T(), outputErrors)

	// backpropogate
	net.outputWeights = helpers.Add(net.outputWeights,
		helpers.Scale(net.learningRate,
			helpers.Dot(helpers.Multiply(outputErrors, helpers.SigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = helpers.Add(net.hiddenWeights,
		helpers.Scale(net.learningRate,
			helpers.Dot(helpers.Multiply(hiddenErrors, helpers.SigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}

func save(net Network) {
	h, err := os.Create("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

// load a neural network from file
func load(net *Network) {
	h, err := os.Open("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}

func mnistTrain(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	for epochs := 0; epochs < 5; epochs++ {
		testFile, _ := os.Open("mnist_dataset/mnist_train.csv")
		r := csv.NewReader(bufio.NewReader(testFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.01
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99

			net.Train(inputs, targets)
		}
		testFile.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
}

func mnistPredict(net *Network) {
	t1 := time.Now()
	checkFile, _ := os.Open("mnist_dataset/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	tests := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}
		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
		tests++
	}

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Printf("Tests run: %d\n", tests)
	fmt.Println("score:", score)
}

func main() {
	// 784 inputs - 28 x 28 pixels, each pixel is an input
	// 200 hidden neurons - an arbitrary number
	// 10 outputs - digits 0 to 9
	// 0.1 is the learning rate
	net := CreateNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	flag.Parse()

	// train or mass predict to determine the effectiveness of the trained network
	switch *mnist {
	case "train":
		mnistTrain(&net)
		save(net)
	case "predict":
		load(&net)
		mnistPredict(&net)
	default:
		// don't do anything
	}
}
