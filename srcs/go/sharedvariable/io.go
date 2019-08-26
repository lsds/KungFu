package sharedvariable

import (
	"io"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
)

func readN(r io.Reader, buffer []byte, n int) error {
	for offset := 0; offset < n; {
		n, err := r.Read(buffer[offset:])
		offset += n
		if err != nil {
			if err == io.EOF {
				if offset == n {
					return nil
				}
				return io.ErrUnexpectedEOF
			}
			return err
		}
	}
	return nil
}

func readBuf(r io.Reader, b *kb.Buffer) error {
	return readN(r, b.Data, len(b.Data))
}

func writeTo(w io.Writer, b *kb.Buffer) error {
	_, err := w.Write(b.Data)
	return err
}
