package rchannel

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

type ConnType uint16

const (
	ConnControl    ConnType 	   = 0
	ConnCollective ConnType 	   = 1
	ConnPeerToPeer ConnType        = 2
	ConnRequestPeerToPeer ConnType = 3
	ConnReplyPeerToPeer ConnType   = 4

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

type messageHeader struct {
	NameLength uint32
	Name       []byte
	BodyInShm  uint32
}

func (h messageHeader) WriteTo(w io.Writer) error {
	if err := binary.Write(w, endian, h.NameLength); err != nil {
		return err
	}
	if _, err := w.Write(h.Name); err != nil {
		return err
	}
	if err := binary.Write(w, endian, h.BodyInShm); err != nil {
		return err
	}
	return nil
}

func (h *messageHeader) ReadFrom(r io.Reader) error {
	if err := binary.Read(r, endian, &h.NameLength); err != nil {
		return err
	}
	h.Name = make([]byte, h.NameLength)
	if err := readN(r, h.Name, int(h.NameLength)); err != nil {
		return err
	}
	if err := binary.Read(r, endian, &h.BodyInShm); err != nil {
		return err
	}
	return nil
}

func (h messageHeader) String() string {
	return fmt.Sprintf("messageHeader{length=%d,name=%s}", h.NameLength, string(h.Name))
}

// messageTail will be sent when messageHeader.BodyInShm != 0.
type messageTail struct {
	Offset uint32
	Length uint32
}

func (m messageTail) WriteTo(w io.Writer) error {
	if err := binary.Write(w, endian, m.Offset); err != nil {
		return err
	}
	if err := binary.Write(w, endian, m.Length); err != nil {
		return err
	}
	return nil
}

func (m *messageTail) ReadFrom(r io.Reader) error {
	if err := binary.Read(r, endian, &m.Offset); err != nil {
		return err
	}
	if err := binary.Read(r, endian, &m.Length); err != nil {
		return err
	}
	return nil
}

// Message is the data transferred via channel
type Message struct {
	From   uint32
	Length uint32
	Data   []byte
}

func (m Message) WriteTo(w io.Writer) error {
	if err := binary.Write(w, endian, m.From); err != nil {
		return err
	}
	if err := binary.Write(w, endian, m.Length); err != nil {
		return err
	}
	_, err := w.Write(m.Data)
	return err
}

func (m *Message) ReadFrom(r io.Reader) error {
	if err := binary.Read(r, endian, &m.From); err != nil {
		return err
	}
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