package gonet

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"math/rand"
	"os"
	"sync"
)

// UTILITIES
func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func matrix(I, J int) [][]float64 {
	m := make([][]float64, I)
	for i := 0; i < I; i++ {
		m[i] = make([]float64, J)
	}
	return m
}

func vector(I int, fill float64) []float64 {
	v := make([]float64, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func linear(x float64) float64 {
	return x
}

func dlinear(y float64) float64 {
	return 1
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func drelu(y float64) float64 {
	if y > 0 {
		return 1
	}
	return 0
}

// SAVE AND RESTORE
var lock sync.Mutex

func marshal(v interface{}) (io.Reader, error) {
	b, err := json.MarshalIndent(v, "", "\t")
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(b), nil
}

func unmarshal(r io.Reader, v interface{}) error {
	return json.NewDecoder(r).Decode(v)
}

// Save neural network to file
func (nn *NN) Save(path string) error {
	lock.Lock()
	defer lock.Unlock()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r, err := marshal(nn)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, r)
	return err
}

// Load neural network from file
func Load(path string) (NN, error) {
	lock.Lock()
	defer lock.Unlock()

	nn := NN{}
	f, err := os.Open(path)
	if err != nil {
		return nn, err
	}
	defer f.Close()
	err = unmarshal(f, &nn)
	return nn, err
}
