package iostream

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// Tee redirects r to ws
func Tee(r io.Reader, ws ...io.Writer) error {
	reader := bufio.NewReader(r)
	for {
		line, _, err := reader.ReadLine()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		for _, w := range ws {
			fmt.Fprintln(w, string(line))
			datas := strings.Split(string(line), ":")
			if string(datas[0]) == "some machine died"{
				return fmt.Errorf(string(line))
			}
		}
	}
}
