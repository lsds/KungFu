package configserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"time"

	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type Client struct {
	endpoint string
	client   http.Client
}

func NewClient(endpoint string) *Client {
	return &Client{endpoint: endpoint}
}

func (cc *Client) Reset() error {
	u, err := url.Parse(cc.endpoint)
	if err != nil {
		return err
	}
	u.Path = `/reset`
	resp, err := cc.client.Post(u.String(), "application/json", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

func (cc *Client) Update(cluster plan.Cluster) error {
	u, err := url.Parse(cc.endpoint)
	if err != nil {
		return err
	}
	u.Path = `/put`
	var body bytes.Buffer
	if err := json.NewEncoder(&body).Encode(cluster); err != nil {
		return err
	}
	resp, err := cc.client.Post(u.String(), "application/json", &body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}
	return nil
}

func (cc *Client) WaitServer() error {
	u, err := url.Parse(cc.endpoint)
	if err != nil {
		return err
	}
	u.Path = `/ping`
	for {
		resp, err := cc.client.Get(u.String())
		if err == nil {
			resp.Body.Close()
			break
		}
		log.Warnf("server is not ready: %v", err)
		time.Sleep(2000 * time.Millisecond)
	}
	return nil
}

func (cc *Client) StopServer() error {
	u, err := url.Parse(cc.endpoint)
	if err != nil {
		return err
	}
	u.Path = `/stop`
	resp, err := cc.client.Get(u.String())
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}
