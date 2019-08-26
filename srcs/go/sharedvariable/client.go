package sharedvariable

import (
	"bytes"
	"errors"
	"log"
	"net/http"
	"net/url"
	"strconv"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

type Client struct {
	endpoint string
	client   http.Client
}

func NewClient(endpoint string) *Client {
	return &Client{
		endpoint: endpoint,
	}
}

func (c *Client) Create(name string, count int, dtype kb.KungFu_Datatype) error {
	u := c.makeURL(name, count, dtype)
	req, err := http.NewRequest(http.MethodPost, u.String(), nil)
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
	return nil
}

func (c *Client) Get(name string, buf *kb.Buffer) error {
	u := c.makeURL(name, buf.Count, buf.Type)
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
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
	return readBuf(resp.Body, buf)
}

func (c *Client) Put(name string, buf *kb.Buffer) error {
	u := c.makeURL(name, buf.Count, buf.Type)
	log.Printf("Client::Put with %d", len(buf.Data))
	b := bytes.NewBuffer(buf.Data)
	req, err := http.NewRequest(http.MethodPut, u.String(), b)
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
	return nil
}

func (c *Client) Add(name string, buf *kb.Buffer) error {
	u := c.makeURL(name, buf.Count, buf.Type)
	b := bytes.NewBuffer(buf.Data)
	req, err := http.NewRequest(http.MethodPatch, u.String(), b)
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
	return nil
}

func (c *Client) makeURL(name string, count int, dtype kb.KungFu_Datatype) url.URL {
	q := &url.Values{}
	q.Set("name", name)
	q.Set("count", strconv.Itoa(count))
	q.Set("dtype", strconv.Itoa(int(dtype)))
	return url.URL{
		Scheme:   `http`,
		Host:     c.endpoint,
		Path:     `/shv/`,
		RawQuery: q.Encode(),
	}
}
