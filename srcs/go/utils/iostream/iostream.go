package iostream

import (
	"bufio"
	"fmt"
	"io"

	"github.com/lsds/KungFu/srcs/go/log"
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

type StreamWatcher struct {
	name    string
	verbose bool
	done    chan struct{}

	history []string

	historyLimit  int
	historyMargin int
}

func NewStreamWatcher(name string, verbose bool) *StreamWatcher {
	return &StreamWatcher{
		name:          name,
		verbose:       verbose,
		done:          make(chan struct{}, 1),
		historyLimit:  1000,
		historyMargin: 100,
	}
}

func (w *StreamWatcher) Watch(r io.Reader) {
	defer func() { w.done <- struct{}{} }()
	reader := bufio.NewReader(r)
	for {
		line, _, err := reader.ReadLine()
		if err != nil {
			if err != io.EOF {
				log.Errorf("pip [%s] end with error: %v", w.name, err)
			}
			return
		}
		if w.verbose {
			log.Infof("[%s] %s", w.name, line)
		}
		w.history = append(w.history, string(line))
		if len(w.history) >= w.historyLimit+w.historyMargin {
			w.history = w.history[w.historyMargin:]
		}
	}
}

func (w *StreamWatcher) Wait() []string {
	<-w.done
	return w.history
}
