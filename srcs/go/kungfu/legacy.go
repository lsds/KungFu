package kungfu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"github.com/lsds/KungFu/srcs/go/plan"
)

func (kf *Kungfu) getCurrentCluster() plan.Cluster {
	kf.Lock()
	defer kf.Unlock()
	return kf.currentCluster.Clone()
}

func (kf *Kungfu) ProposeNewSize(newSize int) error {
	cluster := kf.getCurrentCluster()
	newCluster, err := cluster.Resize(newSize)
	if err != nil {
		return err
	}
	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(newCluster); err != nil {
		return err
	}
	u, err := url.Parse(kf.configServerURL)
	if err != nil {
		return err
	}
	u.Path = `/put`
	req, err := http.NewRequest(http.MethodPut, u.String(), buf)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("KungFu Peer: %s", kf.self))
	resp, err := kf.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}
