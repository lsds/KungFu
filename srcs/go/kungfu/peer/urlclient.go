package peer

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
)

func (p *Peer) openHTTP(client *http.Client, url string) (io.ReadCloser, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", fmt.Sprintf("KungFu Peer: %s", p.self))
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(resp.Status)
	}
	return resp.Body, nil
}

var parseURL = url.Parse
var errUnsupportedURL = errors.New("unsupported URL")

func (p *Peer) openURL(url string) (io.ReadCloser, error) {
	u, err := parseURL(url)
	if err != nil {
		return nil, err
	}
	switch u.Scheme {
	case "http":
		return p.openHTTP(&p.httpClient, url)
	case "https":
		return p.openHTTP(&p.httpClient, url)
	case "file":
		// ignore u.Host
		return os.Open(u.Path)
	}
	return nil, errUnsupportedURL
}
