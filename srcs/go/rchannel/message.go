package rchannel

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

type ConnType uint16

const (
	ConnPing       ConnType = iota // 0
	ConnControl    ConnType = iota
	ConnCollective ConnType = iota
	ConnPeerToPeer ConnType = iota
)

var endian = binary.LittleEndian

var errUnexpectedEnd = errors.New("Unexpected End")

type connectionHeader struct {
	Type    uint16
	SrcPort uint16
	SrcIPv4 uint32
}

func (h connectionHeader) WriteTo(w io.Writer) error {
	return binary.Write(w, endian, &h)
}

func (h *connectionHeader) ReadFrom(r io.Reader) error {
	return binary.Read(r, endian, h)
}

const NoFlag uint32 = 0

const (
	WaitRecvBuf = 1 << iota // The recevier should wait receive buffer
)

type messageHeader struct {
	NameLength uint32
	Name       []byte
	Flags      uint32
}

func (h *messageHeader) HasFlag(flag uint32) bool {
	return h.Flags&flag == flag
}

func (h *messageHeader) WriteTo(w io.Writer) error {
	if err := binary.Write(w, endian, h.NameLength); err != nil {
		return err
	}
	if _, err := w.Write(h.Name); err != nil {
		return err
	}
	if err := binary.Write(w, endian, h.Flags); err != nil {
		return err
	}
	return nil
}

// ReadFrom reads the messageHeader from a reader into new buffer.
// The name length is obtained from the reader and should be trusted.
func (h *messageHeader) ReadFrom(r io.Reader) error {
	if err := binary.Read(r, endian, &h.NameLength); err != nil {
		return err
	}
	h.Name = make([]byte, h.NameLength)
	if err := readN(r, h.Name, int(h.NameLength)); err != nil {
		return err
	}
	if err := binary.Read(r, endian, &h.Flags); err != nil {
		return err
	}
	return nil
}

// ReadFromLike reads the messageHeader from a reader into new buffer.
// The result Name should be checked against hint.
func (h *messageHeader) ReadFromLike(r io.Reader, hint string) error {
	if err := binary.Read(r, endian, &h.NameLength); err != nil {
		return err
	}
	if int(h.NameLength) != len(hint) {
		return fmt.Errorf("unexpected name length: %d", h.NameLength)
	}
	h.Name = make([]byte, h.NameLength)
	if err := readN(r, h.Name, int(h.NameLength)); err != nil {
		return err
	}
	if string(h.Name) != hint {
		return fmt.Errorf("unexpected name %s", h.Name)
	}
	if err := binary.Read(r, endian, &h.Flags); err != nil {
		return err
	}
	return nil
}

func (h messageHeader) String() string {
	return fmt.Sprintf("messageHeader{length=%d,name=%s}", h.NameLength, string(h.Name))
}

// Message is the data transferred via channel
type Message struct {
	Length uint32
	Data   []byte
}

func (m *Message) same(pm *Message) bool {
	return &m.Data[0] == &pm.Data[0]
}

func (m Message) WriteTo(w io.Writer) error {
	if err := binary.Write(w, endian, m.Length); err != nil {
		return err
	}
	_, err := w.Write(m.Data)
	return err
}

// ReadFrom reads the message from a reader into new buffer.
// The message length is obtained from the reader and should be trusted.
func (m *Message) ReadFrom(r io.Reader) error {
	if err := binary.Read(r, endian, &m.Length); err != nil {
		return err
	}
	// m.Data = make([]byte, m.Length)
	m.Data = GetBuf(m.Length) // Use leaky pool
	if err := readN(r, m.Data, int(m.Length)); err != nil {
		return err
	}
	return nil
}

// ReadInto reads the message from a reader into existing buffer.
// The message length obtained from the reader should be checked.
func (m *Message) ReadInto(r io.Reader) error {
	var length uint32
	if err := binary.Read(r, endian, &length); err != nil {
		return err
	}
	if length != m.Length {
		return errors.New("Unexpected message length")
	}
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
