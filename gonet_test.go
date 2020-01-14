package gonet

import (
	"math"
	"testing"
)

func TestPredict(t *testing.T) {
	// AND test
	andTest := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {1}},
	}
	nn := New(2, []int{4}, 1, false)
	nn.Train(andTest, 3000, 0.4, 0.2, false)
	for i := 0; i < len(andTest); i++ {
		input := andTest[i][0]
		output := andTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("AND test failed. Got (%f AND %f) == %f, want %f.", input[0], input[1], math.Round(predict[0]), output[0])
		}
	}

	// OR test
	orTest := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {1}},
	}
	nn.Config(2, []int{4}, 1, false)
	nn.Train(orTest, 3000, 0.4, 0.2, false)
	for i := 0; i < len(orTest); i++ {
		input := orTest[i][0]
		output := orTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("OR test failed. Got (%f OR %f) == %f, want %f.", input[0], input[1], math.Round(predict[0]), output[0])
		}
	}

	// XOR test
	xorTest := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}
	nn.Config(2, []int{4}, 1, false)
	nn.Train(xorTest, 3000, 0.4, 0.2, false)
	for i := 0; i < len(xorTest); i++ {
		input := xorTest[i][0]
		output := xorTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("XOR test failed. Got (%f XOR %f) == %f, want %f.", input[0], input[1], math.Round(predict[0]), output[0])
		}
	}

	// Regression test
	regTest := [][][]float64{
		{{1}, {2}},
		{{2}, {4}},
		{{5}, {10}},
		{{8}, {16}},
	}
	nn.Config(1, []int{3, 3}, 1, true)
	nn.Train(regTest, 3000, 0.6, 0.4, false)
	for i := 0; i < len(regTest); i++ {
		input := regTest[i][0]
		output := regTest[i][1]
		predict := nn.Predict(input)
		if math.Round(predict[0]) != output[0] {
			t.Errorf("Regression test failed. Got %f with input %f, want %f.", math.Round(predict[0]), input[0], output[0])
		}
	}
}
