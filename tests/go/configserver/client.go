package configserver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/utils"
)

type Client struct {
	endpoint string
	client   http.Client
}

func NewClient(endpoint string) *Client {
	return &Client{
		endpoint: endpoint,
		client:   http.Client{Timeout: 1 * time.Second},
	}
}

func (cc *Client) Reset() error {
	resp, err := cc.client.Post(cc.endpoint, "application/json", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func (cc *Client) Update(cluster plan.Cluster) error {
	var body bytes.Buffer
	if err := json.NewEncoder(&body).Encode(cluster); err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPut, cc.endpoint, &body)
	if err != nil {
		return err
	}
	resp, err := cc.client.Do(req)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}
	return nil
}

func (cc *Client) WaitServer() error {
	utils.Poll(context.TODO(), func() bool {
		resp, err := cc.client.Get(cc.endpoint)
		if err == nil {
			resp.Body.Close()
		} else {
			log.Warnf("config server is not ready: %v", err)
		}
		return err == nil
	})
	return nil
}

func (cc *Client) StopServer() error {
	u, err := url.Parse(cc.endpoint)
	if err != nil {
		return err
	}
	u.Path = `/stop`
	resp, err := cc.client.Post(u.String(), "", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}
