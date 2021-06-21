package iostream

import (
	"io"
	"os"
	"sync"
	"strings"
    "strconv"
)

var Std = StdWriters{
	Stdout: os.Stdout,
	Stderr: os.Stderr,
}

type StdReaders struct {
	Stdout io.Reader
	Stderr io.Reader
	Flagdown int
    Epochfinish int
}

type StdWriters struct {
	Stdout io.Writer
	Stderr io.Writer
}

func (r *StdReaders) Stream(ws ...*StdWriters) interface{ Wait() } {
	var outs, errs []io.Writer
	for _, w := range ws {
		outs = append(outs, w.Stdout)
		errs = append(errs, w.Stderr)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		errsig := Tee(r.Stdout, outs...)
		if errsig != nil{
			erros := errsig.Error()
            datas := strings.Split(erros, ":")
			if  datas[0] == "some machine died"{
				epochfi, err := strconv.Atoi(datas[1])
                if err != nil{
                }
                r.Flagdown = 1
                r.Epochfinish = epochfi
			}
		}

		wg.Done()
	}()
	go func() {
		Tee(r.Stderr, errs...)
		wg.Done()

	}()
	return &wg
}

// SaveFirstdWriter remembers the content of the first Write call
type SaveFirstdWriter struct {
	First string
}

func (w *SaveFirstdWriter) Write(bs []byte) (int, error) {
	if len(w.First) == 0 {
		w.First = string(bs)
	}
	return len(bs), nil
}

// Null implements /dev/null
type Null struct{}

func (w *Null) Write(bs []byte) (int, error) {
	return len(bs), nil
}
