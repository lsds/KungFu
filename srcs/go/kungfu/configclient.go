package kungfu

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type ConfigClient struct {
	endpoint string
	client   http.Client
}

func NewDefaultConfigClient() (*ConfigClient, error) {
	configServer := os.Getenv(kc.ConfigServerEnvKey)
	return NewConfigClient(configServer)
}

func NewConfigClient(endpoint string) (*ConfigClient, error) {
	// TODO: parse endpoint
	return &ConfigClient{endpoint: endpoint}, nil
}

func (c *ConfigClient) makeURL(version, name string) url.URL {
	q := url.Values{}
	q.Set("version", version)
	q.Set("name", name)
	return url.URL{
		Scheme:   `http`,
		Host:     c.endpoint,
		RawQuery: q.Encode(),
	}
}

func (c *ConfigClient) makeRequest(method string, u url.URL, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequest(method, u.String(), body)
	if err != nil {
		return nil, err
	}
	req.Header.Add("x-kungfu-peer", os.Getenv(kb.SelfSpecEnvKey))
	return req, nil
}

func (c *ConfigClient) GetConfig(version, name string, i interface{}) error {
	if len(c.endpoint) == 0 {
		val := os.Getenv(name)
		log.Warnf("TODO: get %s from config server %s", name, c.endpoint)
		return plan.FromString(val, i)
	}
	u := c.makeURL(version, name)
	req, err := c.makeRequest(http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}
	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	val := strings.TrimSpace(string(bs))
	return plan.FromString(val, i)
}

func (c *ConfigClient) PutConfig(version, name string, i interface{}) error {
	b := &bytes.Buffer{}
	if err := json.NewEncoder(b).Encode(i); err != nil {
		return err
	}
	u := c.makeURL(version, name)
	req, err := c.makeRequest(http.MethodPost, u, b)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return errors.New(resp.Status)
	}
	defer resp.Body.Close()
	ioutil.ReadAll(resp.Body)
	return nil
}
