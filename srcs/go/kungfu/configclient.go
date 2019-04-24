package kungfu

import (
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

func (c *configClient) getConfig(name string, i interface{}) error {
	if len(c.endpoint) == 0 {
		val := os.Getenv(name)
		log.Warnf("TODO: get %s from config server %s", name, c.endpoint)
		return plan.FromString(val, i)
	}
	q := url.Values{}
	q.Set("name", name)
	u := url.URL{
		Scheme:   `http`,
		Host:     c.endpoint,
		RawQuery: q.Encode(),
	}
	resp, err := c.client.Get(u.String())
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	val := strings.TrimSpace(string(bs))
	log.Infof("got config %s from server %s", val, c.endpoint)
	return plan.FromString(val, i)
}
