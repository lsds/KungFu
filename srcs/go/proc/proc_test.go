package proc

import (
	"fmt"
	"testing"

	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func Test_updatedEnvFrom(t *testing.T) {
	oldEnvs := []string{
		`X=1`,
		`Y=Z=2`,
	}
	newValues := make(Envs)
	newValues[`X`] = "2"
	newEnvs := updatedEnvFrom(newValues, oldEnvs)
	if l := len(newEnvs); l != 2 {
		fmt.Printf("%d: %q\n", l, newEnvs)
	}
	assert.True(len(newEnvs) == 2)
	envMap := parseEnv(newEnvs)
	assert.True(envMap[`X`] == `2`)
	assert.True(envMap[`Y`] == `Z=2`)
}
