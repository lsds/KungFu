package scheduler

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// Proc represents a process
type Proc struct {
	Name    string
	Prog    string
	Args    []string
	Envs    map[string]string
	Host    string
	PubAddr string
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

func updatedEnv(newValues map[string]string) []string {
	envMap := make(map[string]string)
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
