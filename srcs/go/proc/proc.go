package proc

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

type Envs map[string]string

func (e Envs) AddIfMissing(k, v string) {
	if _, ok := e[k]; !ok {
		e[k] = v
	}
}

func Merge(e, f Envs) Envs {
	g := make(Envs)
	for k, v := range e {
		g[k] = v
	}
	for k, v := range f {
		g[k] = v
	}
	return g
}

// Proc represents a general purpose process
type Proc struct {
	Name     string
	Prog     string
	Args     []string
	Envs     Envs
	Hostname string
	LogDir   string
	Dir      string
}

func (p Proc) CmdCtx(ctx context.Context) *exec.Cmd {
	cmd := exec.CommandContext(ctx, p.Prog, p.Args...)
	cmd.Env = updatedEnvFrom(p.Envs, os.Environ())
	cmd.Dir = p.Dir
	return cmd
}

func (p Proc) Script() string {
	buf := &bytes.Buffer{}
	var chdir string
	if len(p.Dir) > 0 {
		chdir = fmt.Sprintf("-C %s", p.Dir)
	}
	fmt.Fprintf(buf, "env %s\\\n", chdir)
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

func updatedEnvFrom(newValues Envs, oldEnvs []string) []string {
	envMap := parseEnv(oldEnvs)
	for k, v := range newValues {
		envMap[k] = v
	}
	var envs []string
	for k, v := range envMap {
		envs = append(envs, fmt.Sprintf("%s=%s", k, v))
	}
	return envs
}

func parseEnv(envs []string) Envs {
	envMap := make(Envs)
	for _, kv := range envs {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) == 2 {
			envMap[parts[0]] = parts[1]
		}
	}
	return envMap
}
