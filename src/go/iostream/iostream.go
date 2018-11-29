package iostream

import (
	"bufio"
	"fmt"
	"io"
	"log"
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

type LogWriter struct {
	Prefix string
}

func (l LogWriter) Write(bs []byte) (int, error) {
	log.Printf("[%s] %s", l.Prefix, string(bs))
	return len(bs), nil
}

func NewLogWriter(name string) io.Writer {
	return &LogWriter{Prefix: name}
}

func LogStream(name string, r io.Reader) error {
	w := NewLogWriter(name)
	return Tee(r, w)
}
