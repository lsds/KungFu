package job

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

type Envs map[string]string

func (e Envs) addIfMissing(k, v string) {
	if _, ok := e[k]; !ok {
		e[k] = v
	}
}

func merge(e, f Envs) Envs {
	g := make(Envs)
	for k, v := range e {
		g[k] = v
	}
	for k, v := range f {
		g[k] = v
	}
	return g
}

// Proc represents a process
type Proc struct {
	Name    string
	Prog    string
	Args    []string
	Envs    Envs
	IPv4    uint32
	PubAddr string
	LogDir  string
}

func (p Proc) Cmd() *exec.Cmd {
	cmd := exec.Command(p.Prog, p.Args...)
	cmd.Env = updatedEnv(p.Envs)
	return cmd
}

func (p Proc) Script() string {
	buf := &bytes.Buffer{}
	fmt.Fprintf(buf, "env \\\n")
	for k, v := range p.Envs {
		fmt.Fprintf(buf, "\t%s=%q \\\n", k, v)
	}
	fmt.Fprintf(buf, "\t%s \\\n", p.Prog)
	for _, a := range p.Args {
		fmt.Fprintf(buf, "\t%s \\\n", a)
	}
	fmt.Fprintf(buf, "\n")
	return buf.String()
}

func updatedEnv(newValues Envs) []string {
	envMap := make(Envs)
	for _, kv := range os.Environ() {
		parts := strings.Split(kv, "=")
		if len(parts) == 2 {
			envMap[parts[0]] = parts[1]
		}
	}
	for k, v := range newValues {
		envMap[k] = v
	}
	var envs []string
	for k, v := range envMap {
		envs = append(envs, fmt.Sprintf("%s=%s", k, v))
	}
	return envs
}
