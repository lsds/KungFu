// from: https://github.com/lgarithm/go/blob/master/net/ssh/ssh.go
// Package ssh is a simple wrapper for golang.org/x/crypto/ssh
package ssh

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"os/user"
	"path"
	"sync"
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

// Run runs a command with no context.
func (c *Client) Run(cmd string) ([]byte, []byte, error) {
	return c.RunWith(context.TODO(), cmd)
}

// RunWith runs a command with context.
func (c *Client) RunWith(ctx context.Context, cmd string) ([]byte, []byte, error) {
	session, err := c.client.NewSession()
	if err != nil {
		return nil, nil, err
	}
	defer session.Close()
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	session.Stdout = &stdout
	session.Stderr = &stderr
	if err := session.Start(cmd); err != nil {
		return nil, nil, err
	}
	done := make(chan struct{})
	go func() {
		err = session.Wait()
		done <- struct{}{}
	}()
	select {
	case <-done:
		return stdout.Bytes(), stderr.Bytes(), err
	case <-ctx.Done():
		return stdout.Bytes(), stderr.Bytes(), ctx.Err()
	}
}

// Stream streams STDOUT and STDERR of a command
func (c *Client) Stream(cmd string) error {
	session, err := c.client.NewSession()
	if err != nil {
		return err
	}
	defer session.Close()
	var wg sync.WaitGroup
	if stdout, err := session.StdoutPipe(); err == nil {
		wg.Add(1)
		go func() { streamPipe("stdout", stdout); wg.Done() }()
	} else {
		return err
	}
	if stderr, err := session.StderrPipe(); err == nil {
		wg.Add(1)
		go func() { streamPipe("stderr", stderr); wg.Done() }()
	} else {
		return err
	}
	if err := session.Start(cmd); err != nil {
		return err
	}
	err = session.Wait()
	wg.Wait()
	return err
}

type Watcher func(r io.Reader)

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
	results := iostream.StdReaders{Stdout: stdout, Stderr: stderr}
	ioDone := results.Stream(redirectors...)
	if err := session.Start(cmd); err != nil {
		return err
	}
	done := make(chan error)
	go func() {
		err := session.Wait()
		ioDone.Wait()
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		session.Close() // doesn't work!
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

// Scp copies a file to remote host
func (c *Client) Scp(file, target string) error {
	info, err := os.Stat(file)
	if err != nil {
		return err
	}
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	str := base64.StdEncoding.EncodeToString(data)
	cmd := fmt.Sprintf("echo '%s' | base64 --decode > %s", str, target)
	_, _, err = c.Run(cmd)
	if err != nil {
		return err
	}
	if _, _, err := c.Run(fmt.Sprintf("chmod %04o %s", info.Mode(), target)); err != nil {
		return err
	}
	return nil
}

// Close closes the client
func (c *Client) Close() error {
	return c.client.Close()
}

func streamPipe(name string, r io.Reader) error {
	reader := bufio.NewReader(r)
	for {
		line, _, err := reader.ReadLine()
		if err != nil {
			if err != io.EOF {
				return err
			}
			break
		}
		fmt.Printf("[%s] %s\n", name, line)
	}
	return nil
}

// RunWith provides a convenient wrapper for running a command once.
func RunWith(ctx context.Context, user, host, cmd string) ([]byte, []byte, error) {
	config := Config{
		User: user,
		Host: host,
	}
	client, err := New(config)
	if err != nil {
		return nil, nil, err
	}
	defer client.Close()
	return client.RunWith(ctx, cmd)
}

func RunScript(user, host string, script string) error {
	cfg := Config{User: user, Host: host}
	cli, err := New(cfg)
	if err != nil {
		return err
	}
	cmd := fmt.Sprintf("echo %s | base64 --decode | sh", base64endode(script))
	return cli.Stream(cmd)
}

func base64endode(src string) string {
	return base64.StdEncoding.EncodeToString([]byte(src))
}
