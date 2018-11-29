package rchannel

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

var endian = binary.LittleEndian

var errUnexpectedEnd = errors.New("Unexpected End")

type connectionHeader struct {
	Port uint32
}

func (h connectionHeader) WriteTo(w io.Writer) error {
	return binary.Write(w, endian, h.Port)
}

func (h *connectionHeader) ReadFrom(r io.Reader) error {
	return binary.Read(r, endian, &h.Port)
}

type messageHeader struct {
	NameLength uint32
	Name       []byte
}

func (h messageHeader) WriteTo(w io.Writer) error {
	if err := binary.Write(w, endian, h.NameLength); err != nil {
		return err
	}
	_, err := w.Write(h.Name)
	return err
}

func (h *messageHeader) ReadFrom(r io.Reader) error {
	if err := binary.Read(r, endian, &h.NameLength); err != nil {
		return err
	}
	h.Name = make([]byte, h.NameLength)
	n, err := r.Read(h.Name)
	if err != nil {
		return err
	}
	if n != int(h.NameLength) {
		return errUnexpectedEnd
	}
	return nil
}

func (mh messageHeader) String() string {
	return fmt.Sprintf("messageHeader{length=%d,name=%s}", mh.NameLength, string(mh.Name))
}

// Message is the data transferred via channel
type Message struct {
	Length uint32
	Data   []byte
}

// NewMessage creates a Message with give payload
func NewMessage(bs []byte) *Message {
	return &Message{
		Length: uint32(len(bs)),
		Data:   bs,
	}
}

func (m Message) WriteTo(w io.Writer) error {
	if err := binary.Write(w, endian, m.Length); err != nil {
		return err
	}
	_, err := w.Write(m.Data)
	return err
}

func (m *Message) ReadFrom(r io.Reader) error {
	if err := binary.Read(r, endian, &m.Length); err != nil {
		return err
	}
	m.Data = make([]byte, m.Length)
	if err := readN(r, m.Data, int(m.Length)); err != nil {
		return err
	}
	return nil
}

func (m Message) String() string {
	return fmt.Sprintf("message{length=%d}", m.Length)
}

func readN(r io.Reader, buffer []byte, n int) error {
	for offset := 0; offset < n; {
		n, err := r.Read(buffer[offset:])
		if err != nil {
			return err
		}
		offset += n
	}
	return nil
}
