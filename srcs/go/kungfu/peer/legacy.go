package peer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"github.com/lsds/KungFu/srcs/go/plan"
)

func (p *Peer) getCurrentCluster() plan.Cluster {
	p.Lock()
	defer p.Unlock()
	return p.currentCluster.Clone()
}

func (p *Peer) ProposeNewSize(newSize int) error {
	cluster := p.getCurrentCluster()
	newCluster, err := cluster.Resize(newSize)
	if err != nil {
		return err
	}
	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(newCluster); err != nil {
		return err
	}
	u, err := url.Parse(p.configServerURL)
	if err != nil {
		return err
	}
	u.Path = `/put`
	req, err := http.NewRequest(http.MethodPut, u.String(), buf)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("KungFu Peer: %s", p.self))
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}
