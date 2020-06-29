package utils

import (
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
)

func openHTTP(client *http.Client, url string, userAgent string) (io.ReadCloser, error) {
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, errors.New(resp.Status)
	}
	return resp.Body, nil
}

var parseURL = url.Parse
var errUnsupportedURL = errors.New("unsupported URL")

// OpenURL opens a file or URL as io.ReadCloser.
func OpenURL(url string, client *http.Client, userAgent string) (io.ReadCloser, error) {
	u, err := parseURL(url)
	if err != nil {
		return nil, err
	}
	switch u.Scheme {
	case "http":
		return openHTTP(client, url, userAgent)
	case "https":
		return openHTTP(client, url, userAgent)
	case "file":
		// ignore u.Host
		return os.Open(u.Path)
	}
	return nil, errUnsupportedURL
}
