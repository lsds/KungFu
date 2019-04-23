package kungfu

import (
	"os"

	kc "github.com/lsds/KungFu/srcs/go/kungfuconfig"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type configClient struct {
	endpoint string
}

func newConfigClient() (*configClient, error) {
	configServer := os.Getenv(kc.ConfigServerEnvKey)
	// TODO: parse configServer
	return &configClient{endpoint: configServer}, nil
}

func (c *configClient) getConfig(name string, i interface{}) error {
	val := os.Getenv(name)
	log.Warnf("TODO: get %s from config server %s", name, c.endpoint)
	return plan.FromString(val, i)
}
