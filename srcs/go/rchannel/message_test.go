package rchannel

import (
	"bytes"
	"testing"
)

func Test_Message(t *testing.T) {
	b := &bytes.Buffer{}
	{
		bs := []byte("123456")
		m := Message{
			Length: uint32(len(bs)),
			Data:   bs,
		}
		if err := m.WriteTo(b); err != nil {
			t.Errorf("Message::WriteTo failed: %v", err)
		}
	}
	{
		var m Message
		if err := m.ReadFrom(b); err != nil {
			t.Errorf("Message::ReadFrom failed: %v", err)
		}

		if m.Length != 6 {
			t.Errorf("Message::ReadFrom unexpected data")
		}
		if string(m.Data) != "123456" {
			t.Errorf("Message::ReadFrom unexpected data")
		}
	}
}

func Test_long_Message(t *testing.T) {
	b := &bytes.Buffer{}
	const str8 = `01234567`
	str1Ki := repeat(128, str8)
	str1Mi := repeat(1024, str1Ki)
	payload := repeat(16, str1Mi)
	{
		bs := []byte(payload)
		m := Message{
			Length: uint32(len(bs)),
			Data:   bs,
		}
		if err := m.WriteTo(b); err != nil {
			t.Errorf("Message::WriteTo failed: %v", err)
		}
	}
	{
		var m Message
		if err := m.ReadFrom(b); err != nil {
			t.Errorf("Message::ReadFrom failed: %v", err)
		}
		if int(m.Length) != len(payload) {
			t.Errorf("Message::ReadFrom unexpected data")
		}
	}
}

func Test_messageHeader(t *testing.T) {
	b := &bytes.Buffer{}
	{
		bs := []byte("123456")
		h := messageHeader{
			NameLength: uint32(len(bs)),
			Name:       bs,
		}
		if err := h.WriteTo(b); err != nil {
			t.Errorf("Message::WriteTo failed: %v", err)
		}
	}
	{
		var h messageHeader
		if err := h.ReadFrom(b); err != nil {
			t.Errorf("Message::ReadFrom failed: %v", err)
		}

		if h.NameLength != 6 {
			t.Errorf("Message::ReadFrom unexpected data")
		}
		if string(h.Name) != "123456" {
			t.Errorf("Message::ReadFrom unexpected data")
		}
	}
}

func repeat(n int, str string) string {
	var ss string
	for i := 0; i < n; i++ {
		ss += str
	}
	return ss
}
