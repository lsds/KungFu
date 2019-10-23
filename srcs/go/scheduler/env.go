package scheduler

import (
	"os/exec"
	"strings"
)

var DefaultLdLibraryPath = GetLdLibraryPath()

func GetLdLibraryPath() string {
	pythonScript := `import os; import kungfu; print(os.path.dirname(kungfu.__file__))`
	cmd := exec.Command(`python3`, `-c`, pythonScript)
	bs, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(bs))
}
