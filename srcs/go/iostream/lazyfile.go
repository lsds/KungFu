package iostream

import (
	"io"
	"os"
)

type layzeFile struct {
	name string
	f    io.WriteCloser
}

func NewLazyFile(filename string) io.WriteCloser {
	// FIXME: check filename
	return &layzeFile{name: filename}
}

func (f *layzeFile) Write(bs []byte) (int, error) {
	if f.f == nil {
		var err error
		if f.f, err = os.Create(f.name); err != nil {
			return 0, err
		}
	}
	return f.f.Write(bs)
}

func (f *layzeFile) Close() error {
	if f.f != nil {
		return f.f.Close()
	}
	return nil
}
