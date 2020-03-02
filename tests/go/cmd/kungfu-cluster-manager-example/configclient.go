package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type configClient struct {
	endpoint string
	client   http.Client
}

func (cc *configClient) Update(cluster plan.Cluster) error {
	var body bytes.Buffer
	if err := json.NewEncoder(&body).Encode(cluster); err != nil {
		return err
	}
	resp, err := cc.client.Post(cc.endpoint, "application/json", &body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}
	return nil
}

func (cc *configClient) Wait() {
	for {
		resp, err := cc.client.Get(cc.endpoint)
		if err == nil {
			resp.Body.Close()
			break
		}
		log.Warnf("server is not ready: %v", err)
		time.Sleep(2000 * time.Millisecond)
	}
}
