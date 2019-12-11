package job

import (
	"errors"
	"os"
	"strconv"
	"strings"

	"github.com/lsds/KungFu/srcs/go/log"
)

// https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
const cudaVisibleDevicesKey = `CUDA_VISIBLE_DEVICES`

var lookupEnv = os.LookupEnv

func getCudaIndex(localRank int) int {
	val, ok := lookupEnv(cudaVisibleDevicesKey)
	if !ok {
		return localRank
	}
	ids, err := parseCudaVisibleDevices(val)
	if err != nil {
		log.Warnf("invalid valud of %s: %q", cudaVisibleDevicesKey, val)
		return -1
	}
	if len(ids) <= localRank {
		log.Warnf("%s=%s is not enough for local rank %d", cudaVisibleDevicesKey, val, localRank)
		return -1
	}
	return ids[localRank]
}

var errInvalidCudaVisibleDevices = errors.New("invalid " + cudaVisibleDevicesKey)

func parseCudaVisibleDevices(val string) ([]int, error) {
	if len(val) == 0 {
		return nil, nil
	}
	parts := strings.Split(val, ",")
	set := make(map[int]struct{})
	var ids []int
	for _, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		if n < 0 {
			continue
		}
		if _, ok := set[n]; ok {
			return nil, errInvalidCudaVisibleDevices
		}
		set[n] = struct{}{}
		ids = append(ids, n)
	}
	return ids, nil
}
