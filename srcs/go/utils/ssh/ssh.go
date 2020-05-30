// from: https://github.com/lgarithm/go/blob/master/net/ssh/ssh.go
// Package ssh is a simple wrapper for golang.org/x/crypto/ssh
package ssh

import (
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"os/user"
	"path"
	"time"

	"github.com/lsds/KungFu/srcs/go/utils/iostream"
	"golang.org/x/crypto/ssh"
)

var defaultTimeout = 8 * time.Second

// Config is a pair of user and host
type Config struct {
	User string
	Host string
}

func withDefaultPort(host string) string {
	_, _, err := net.SplitHostPort(host)
	if err == nil {
		return host
	}
	const defaultPort = "22"
	return net.JoinHostPort(host, defaultPort)
}

func withDefaultUser(name string) string {
	if len(name) == 0 {
		if u, err := user.Current(); err == nil {
			return u.Username
		}
	}
	return name
}

func completeConfig(config Config) Config {
	return Config{
		User: withDefaultUser(config.User),
		Host: withDefaultPort(config.Host),
	}
}

func newSSHClient(config Config) (*ssh.Client, error) {
	config = completeConfig(config)
	key, err := defaultKeyFile()
	if err != nil {
		return nil, errors.New("failed to get key")
	}
	clientConfig := &ssh.ClientConfig{
		User: config.User,
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(key),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         defaultTimeout,
	}
	client, err := ssh.Dial("tcp", config.Host, clientConfig)
	if err != nil {
		return nil, err
	}
	return client, nil
}

// Client is a wrapper for ssh.Client
type Client struct {
	config Config
	client *ssh.Client
}

// New creates a new Client
func New(cfg Config) (*Client, error) {
	client, err := newSSHClient(cfg)
	if err != nil {
		return nil, err
	}
	return &Client{cfg, client}, err
}

func (c *Client) String() string {
	return fmt.Sprintf("%s@%s", c.config.User, c.config.Host)
}

func (c *Client) Watch(ctx context.Context, cmd string, redirectors []*iostream.StdWriters) error {
	session, err := c.client.NewSession()
	if err != nil {
		return err
	}
	defer session.Close()
	stdout, err := session.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := session.StderrPipe()
	if err != nil {
		return err
	}
	if err := session.RequestPty("xterm", 80, 40, nil); err != nil {
		return err
	}
	results := iostream.StdReaders{Stdout: stdout, Stderr: stderr}
	ioDone := results.Stream(redirectors...)
	if err := session.Start(cmd); err != nil {
		return err
	}
	done := make(chan error)
	go func() {
		ioDone.Wait() // before session.Wait()
		err := session.Wait()
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		session.Close()
		// FIXME: force remote command terminate
		return ctx.Err()
	}
}

func defaultKeyFile() (ssh.Signer, error) {
	usr, _ := user.Current()
	file := path.Join(usr.HomeDir, ".ssh", "id_rsa")
	buf, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return ssh.ParsePrivateKey(buf)
}

// Close closes the client
func (c *Client) Close() error {
	return c.client.Close()
}
