package main

import "C"
import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"os"
	"reflect"
	"unsafe"

	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

var httpClient http.Client

func readConfigServer(url string) (*plan.Cluster, error) {
	f, err := utils.OpenURL(url, &httpClient, "")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var cluster plan.Cluster
	if err = json.NewDecoder(f).Decode(&cluster); err != nil {
		return nil, err
	}
	return &cluster, nil
}

//export GoReadConfigServer
func GoReadConfigServer(ptr unsafe.Pointer, ptrSize int) int {
	cluster, err := readConfigServer(os.Getenv(`KUNGFU_CONFIG_SERVER`))
	if err != nil {
		return -1
	}
	*(*uint32)(unsafe.Pointer(uintptr(ptr) + 0)) = uint32(len(cluster.Runners))
	*(*uint32)(unsafe.Pointer(uintptr(ptr) + 4)) = uint32(len(cluster.Workers))
	out := goBytes(ptr, ptrSize)
	bs := cluster.Bytes()
	copy(out[8:8+len(bs)], bs)
	return len(bs) + 8
}

func parseCluster(ptr unsafe.Pointer, ptrSize int) plan.Cluster {
	nr := *(*uint32)(unsafe.Pointer(uintptr(ptr) + 0))
	nw := *(*uint32)(unsafe.Pointer(uintptr(ptr) + 4))
	cluster := plan.Cluster{
		Runners: make(plan.PeerList, nr),
		Workers: make(plan.PeerList, nw),
	}
	out := goBytes(ptr, ptrSize)
	br := bytes.NewReader(out[8:])
	for i := range cluster.Runners {
		binary.Read(br, binary.LittleEndian, &cluster.Runners[i])
	}
	for i := range cluster.Workers {
		binary.Read(br, binary.LittleEndian, &cluster.Workers[i])
	}
	return cluster
}

func main() {}

func goBytes(ptr unsafe.Pointer, ptrSize int) []byte {
	sh := &reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  ptrSize,
		Cap:  ptrSize,
	}
	return *(*[]byte)(unsafe.Pointer(sh))
}
