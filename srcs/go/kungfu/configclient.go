package kungfu

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type configClient struct {
	endpoint string
	client   http.Client
}

func newConfigClient() (*configClient, error) {
	configServer := os.Getenv(kc.ConfigServerEnvKey)
	// TODO: parse configServer
	return &configClient{endpoint: configServer}, nil
}

func (c *configClient) makeURL(version, name string) url.URL {
	q := url.Values{}
	q.Set("version", version)
	q.Set("name", name)
	return url.URL{
		Scheme:   `http`,
		Host:     c.endpoint,
		RawQuery: q.Encode(),
	}
}

func (c *configClient) getConfig(version, name string, i interface{}) error {
	if len(c.endpoint) == 0 {
		val := os.Getenv(name)
		log.Warnf("TODO: get %s from config server %s", name, c.endpoint)
		return plan.FromString(val, i)
	}
	u := c.makeURL(version, name)
	resp, err := c.client.Get(u.String())
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

func (c *configClient) putConfig(version, name string, i interface{}) error {
	b := &bytes.Buffer{}
	if err := json.NewEncoder(b).Encode(i); err != nil {
		return err
	}
	u := c.makeURL(version, name)
	resp, err := c.client.Post(u.String(), "application/json", b)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusCreated {
		return errors.New(resp.Status)
	}
	defer resp.Body.Close()
	ioutil.ReadAll(resp.Body)
	return nil
}
