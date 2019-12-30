package iostream

import (
	"bufio"
	"fmt"
	"io"
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
		}
	}
}
