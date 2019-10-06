package log

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfuconfig"
)

var std = New()

type Logger struct {
	sync.Mutex
	w     io.Writer
	buf   []byte
	t0    time.Time
	debug bool
}

func New() *Logger {
	l := &Logger{
		w:     os.Stdout,
		t0:    time.Now(),
		debug: kungfuconfig.ShowDebugLog,
	}
	return l
}

func fmtDuration(d time.Duration) string {
	n := int64(d / time.Second)

	ss := n % 60
	n /= 60

	mm := n % 60
	n /= 60

	hh := n % 24
	n /= 24

	ns := int64(d % time.Second)

	return fmt.Sprintf("%dd %02d:%02d:%02d %6.2fms", n, hh, mm, ss, float64(ns)/float64(time.Millisecond))
}

func (l *Logger) output(prefix, format string, v ...interface{}) {
	l.Lock()
	defer l.Unlock()
	// d := time.Since(l.t0)
	l.buf = l.buf[:0]
	l.buf = append(l.buf, prefix...)
	// l.buf = append(l.buf, ' ', '[')
	// l.buf = append(l.buf, fmtDuration(d)...)
	// l.buf = append(l.buf, ']', ' ')
	s := fmt.Sprintf(format, v...)
	l.buf = append(l.buf, s...)
	if len(s) == 0 || s[len(s)-1] != '\n' {
		l.buf = append(l.buf, '\n')
	}
	l.w.Write(l.buf)
}

func (l *Logger) logf(level, format string, v ...interface{}) {
	l.output(level, format, v...)
}

func (l *Logger) Debugf(format string, v ...interface{}) {
	if l.debug {
		l.logf("[D] ", format, v...)
	}
}

func (l *Logger) Infof(format string, v ...interface{}) {
	l.logf("[I] ", format, v...)
}

func (l *Logger) Warnf(format string, v ...interface{}) {
	l.logf("[W] ", format, v...)
}

func (l *Logger) Errorf(format string, v ...interface{}) {
	l.logf("[E] ", format, v...)
}

func (l *Logger) Exitf(format string, v ...interface{}) {
	l.logf("[E] ", format, v...)
	os.Exit(1)
}

var (
	Debugf = std.Debugf
	Infof  = std.Infof
	Warnf  = std.Warnf
	Errorf = std.Errorf
	Exitf  = std.Exitf
)
