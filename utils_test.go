package gonet

import (
	"os"
	"testing"
)

func TestUtils(t *testing.T) {
	nn := New(2, []int{4}, 1, false)

	nn.Save("model.json")
	defer os.Remove("model.json")

	nn, err := Load("model.json")
	if err != nil {
		t.Errorf("Utils test failed.")
	}
}
