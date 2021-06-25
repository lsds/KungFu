package env

import "encoding/json"

type peer struct {
}

type kungfuConfig struct {
	Peers []peer `json:"peers"`
}

func ParseConfigFromJSON(js string) (*Config, error) {
	var kfConfig kungfuConfig
	if err := json.Unmarshal([]byte(js), &kfConfig); err != nil {
		return nil, err
	}
	return &Config{
		// InitPeers: initPeers,
		// Strategy:  *strategy,
	}, nil
}
