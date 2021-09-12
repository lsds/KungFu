package main

import "C"
import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

func writeConfig(cluster *plan.Cluster, url string) error {
	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(cluster); err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPut, url, buf)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("KungFu Runner"))
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func propose(n int) error {
	url := os.Getenv(`KUNGFU_CONFIG_SERVER`)
	cluster, err := readConfigServer(url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "readConfigServer failed: %v\n", err)
		return err
	}
	fmt.Fprintf(os.Stderr, "old cluster: %s\n", cluster)
	newCluster, err := cluster.Resize(n)
	fmt.Fprintf(os.Stderr, "new cluster: %s\n", newCluster)
	if err != nil {
		return err
	}
	return writeConfig(newCluster, url)
}

//export GoPropose
func GoPropose(n int) {
	fmt.Fprintf(os.Stderr, "[go] propose new size %d\n", n)
	if err := propose(n); err != nil {
		log.Warnf("propose(%d) failed: %v", n, err)
	}
	fmt.Fprintf(os.Stderr, "[go] proposed new size %d\n", n)
}
