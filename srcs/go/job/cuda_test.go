package job

import (
	"os"
	"testing"
)

var cudaEnv = map[string]string{}

func mockLookupEnv(key string) (string, bool) {
	val, ok := cudaEnv[key]
	return val, ok
}

func Test_getCudaIndex(t *testing.T) {
	lookupEnv = mockLookupEnv
	defer func() { lookupEnv = os.LookupEnv }()

	if id := getCudaIndex(1); id != 1 {
		t.Errorf("want %d, got %d", 1, id)
	}

	cudaEnv[`CUDA_VISIBLE_DEVICES`] = "2,3"
	if id := getCudaIndex(1); id != 3 {
		t.Errorf("want %d, got %d", 3, id)
	}

	cudaEnv[`CUDA_VISIBLE_DEVICES`] = ""
	if id := getCudaIndex(0); id != -1 {
		t.Errorf("want %d, got %d", -1, id)
	}
}
