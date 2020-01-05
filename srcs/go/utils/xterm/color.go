package xterm

import (
	"bytes"
	"fmt"
)

type ColorSet []Color

func (cs ColorSet) Choose(i int) Color {
	return cs[i%len(cs)]
}

var (
	BasicColors = ColorSet{
		Green,
		Blue,
		Yellow,
		LightBlue,
	}

	Warn = Red
)

type Color interface {
	B(text string) []byte
	S(text string) string
}

type color struct {
	f uint8
	b uint8
}

// Standard XTerm Colors
var (
	Green     = color{f: 32, b: 1}
	Yellow    = color{f: 33, b: 1}
	Blue      = color{f: 34, b: 1}
	Red       = color{f: 35, b: 1}
	LightBlue = color{f: 36, b: 1}
	Grey      = color{f: 37, b: 1}
)

func (c color) bs(text string) *bytes.Buffer {
	buf := &bytes.Buffer{}
	fmt.Fprintf(buf, "\x1b[%d;%dm", c.b, c.f)
	buf.WriteString(text)
	buf.WriteString("\x1b[m")
	return buf
}

func (c color) B(text string) []byte {
	return c.bs(text).Bytes()
}

func (c color) S(text string) string {
	return c.bs(text).String()
}

var NoColor = noColor{}

type noColor struct{}

func (c noColor) B(text string) []byte {
	return []byte(text)
}

func (c noColor) S(text string) string {
	return text
}
