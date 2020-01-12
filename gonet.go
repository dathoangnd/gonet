package gonet

import (
	"log"
	"math"
	"math/rand"
)

// NN struct is used to represent a neural network
type NN struct {
	// Whether the problem is regression or classification
	Regression bool
	// Number of nodes in each layer
	NNodes []int
	// Activations for each layer
	Activations [][]float64
	// Weights
	Weights [][][]float64
	// Last change in weights for momentum
	Changes [][][]float64
}

/*
New creates a new neural network
'nInputs' is number of nodes in input layer
'nHiddens' is array of numbers of nodes in hidden layers
'nOutputs' is number of nodes in output layer
'isRegression' is whether the problem is regression or classification
*/
func New(nInputs int, nHiddens []int, nOutputs int, isRegression bool) NN {
	nn := NN{}
	nn.Config(nInputs, nHiddens, nOutputs, isRegression)
	return nn
}

/*
Config the neural network, also reset all trained weights
'nInputs' is number of nodes in input layer
'nHiddens' is array of numbers of nodes in hidden layers
'nOutputs' is number of nodes in output layer
'isRegression' is whether the problem is regression or classification
*/
func (nn *NN) Config(nInputs int, nHiddens []int, nOutputs int, isRegression bool) {
	if len(nHiddens) == 0 {
		log.Fatal("Should have at least 1 hidden layer")
	}

	nn.Regression = isRegression

	nn.NNodes = []int{
		nInputs + 1, // +1 for bias
	}
	for i := 0; i < len(nHiddens); i++ {
		nn.NNodes = append(nn.NNodes, nHiddens[i]+1) // +1 for bias
	}
	nn.NNodes = append(nn.NNodes, nOutputs)

	NLayers := len(nn.NNodes)

	nn.Activations = make([][]float64, 0)
	for i := 0; i < NLayers; i++ {
		nn.Activations = append(nn.Activations, vector(nn.NNodes[i], 1.0))
	}
	nn.Weights = make([][][]float64, NLayers-1)
	nn.Changes = make([][][]float64, NLayers-1)

	for i := 0; i < len(nn.Weights); i++ {
		nn.Weights[i] = matrix(nn.NNodes[i], nn.NNodes[i+1])
		nn.Changes[i] = matrix(nn.NNodes[i], nn.NNodes[i+1])
	}

	rand.Seed(0)
	for i := 0; i < len(nn.Weights); i++ {
		for j := 0; j < len(nn.Weights[i]); j++ {
			for k := 0; k < len(nn.Weights[i][j]); k++ {
				nn.Weights[i][j][k] = random(-1, 1)
			}
		}
	}
}

// Feed forward the neural network
func (nn *NN) feedForward(inputs []float64) []float64 {
	NLayers := len(nn.NNodes)
	if NLayers < 3 {
		log.Fatal("Should have at least 1 hidden layer")
	}

	if len(inputs) != nn.NNodes[0]-1 {
		log.Fatal("Error: wrong number of inputs")
	}
	for i := 0; i < nn.NNodes[0]-1; i++ {
		nn.Activations[0][i] = inputs[i]
	}

	for k := 1; k < NLayers-1; k++ {
		for i := 0; i < nn.NNodes[k]-1; i++ {
			var sum float64

			for j := 0; j < nn.NNodes[k-1]; j++ {
				sum += nn.Activations[k-1][j] * nn.Weights[k-1][j][i]
			}

			if nn.Regression {
				// Use sigmoid to avoid explosion
				nn.Activations[k][i] = sigmoid(sum)
			} else {
				nn.Activations[k][i] = relu(sum)
			}
		}
	}

	for i := 0; i < nn.NNodes[NLayers-1]; i++ {
		var sum float64

		for j := 0; j < nn.NNodes[NLayers-2]; j++ {
			sum += nn.Activations[NLayers-2][j] * nn.Weights[NLayers-2][j][i]
		}

		if nn.Regression {
			nn.Activations[NLayers-1][i] = linear(sum)
		} else {
			nn.Activations[NLayers-1][i] = sigmoid(sum)
		}
	}

	return nn.Activations[NLayers-1]
}

/*
Update weights with Back Propagation algorithm
'targets' is traning outputs
'lRate' is learning rate
'mFactor' is used by momentum gradient discent
return the prediction error
*/
func (nn *NN) backPropagate(targets []float64, lRate, mFactor float64) float64 {
	NLayers := len(nn.NNodes)
	if NLayers < 3 {
		log.Fatal("Should have at least 1 hidden layer")
	}

	if len(targets) != nn.NNodes[NLayers-1] {
		log.Fatal("Error: wrong number of target values")
	}

	deltas := make([][]float64, NLayers-1)
	deltas[NLayers-2] = vector(nn.NNodes[NLayers-1], 0.0)
	for i := 0; i < nn.NNodes[NLayers-1]; i++ {
		if nn.Regression {
			deltas[NLayers-2][i] = dlinear(nn.Activations[NLayers-1][i]) * (targets[i] - nn.Activations[NLayers-1][i])
		} else {
			deltas[NLayers-2][i] = dsigmoid(nn.Activations[NLayers-1][i]) * (targets[i] - nn.Activations[NLayers-1][i])
		}
	}

	for k := len(deltas) - 2; k >= 0; k-- {
		deltas[k] = vector(nn.NNodes[k+1], 0.0)
		for i := 0; i < nn.NNodes[k+1]; i++ {
			var e float64

			for j := 0; j < nn.NNodes[k+2]-1; j++ {
				e += deltas[k+1][j] * nn.Weights[k+1][i][j]
			}

			if nn.Regression {
				deltas[k][i] = dsigmoid(nn.Activations[k+1][i]) * e
			} else {
				deltas[k][i] = drelu(nn.Activations[k+1][i]) * e
			}
		}
	}

	for k := NLayers - 2; k >= 0; k-- {
		for i := 0; i < nn.NNodes[k]; i++ {
			for j := 0; j < nn.NNodes[k+1]; j++ {
				change := deltas[k][j] * nn.Activations[k][i]
				nn.Weights[k][i][j] = nn.Weights[k][i][j] + lRate*change + mFactor*nn.Changes[k][i][j]
				nn.Changes[k][i][j] = change
			}
		}
	}
	var err float64
	for i := 0; i < len(targets); i++ {
		err += 0.5 * math.Pow(targets[i]-nn.Activations[NLayers-1][i], 2)
	}
	return err
}

/*
Train the neural network
'inputs' is the training data
'iterations' is the number to run feed forward and back propagation
'lRate' is learning rate
'mFactor' is used by momentum gradient discent
return errors track lists during training time
*/
func (nn *NN) Train(inputs [][][]float64, iterations int, lRate, mFactor float64) []float64 {
	errors := make([]float64, 0)

	for i := 0; i < iterations; i++ {
		var e float64
		for _, p := range inputs {
			nn.feedForward(p[0])

			tmp := nn.backPropagate(p[1], lRate, mFactor)
			e += tmp
		}
		if i%1000 == 0 {
			errors = append(errors, e)
		}
	}

	return errors
}

/*
Predict output with new input
*/
func (nn *NN) Predict(input []float64) []float64 {
	return nn.feedForward(input)
}
