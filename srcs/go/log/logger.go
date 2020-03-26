package log

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/lsds/KungFu/srcs/go/kungfu/config"
	"github.com/lsds/KungFu/srcs/go/utils/xterm"
)

type Level int32

const (
	Debug Level = iota
	Info  Level = iota
	Warn  Level = iota
	Error Level = iota
)

var logLevelMap = map[string]Level{
	`DEBUG`: Debug,
	`INFO`:  Info,
	`WARN`:  Warn,
	`ERROR`: Error,
}

func parseLogLevel(val string) Level {
	return logLevelMap[val]
}

var std = New()

const (
	ShowTimestamp = 1 << iota
)

type Logger struct {
	sync.Mutex
	outWriter io.Writer
	errWriter io.Writer
	buf       []byte
	t0        time.Time
	level     Level
	flags     uint32
}

func New() *Logger {
	l := &Logger{
		outWriter: os.Stdout,
		errWriter: os.Stderr,
		t0:        time.Now(),
		level:     parseLogLevel(config.LogLevel),
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

func (l *Logger) output(w io.Writer, prefix, format string, v ...interface{}) {
	l.Lock()
	defer l.Unlock()
	d := time.Since(l.t0)
	l.buf = l.buf[:0]
	l.buf = append(l.buf, prefix...)
	if l.flags&ShowTimestamp != 0 {
		l.buf = append(l.buf, ' ', '[')
		l.buf = append(l.buf, fmtDuration(d)...)
		l.buf = append(l.buf, ']', ' ')
	} else {
		l.buf = append(l.buf, ' ')
	}
	s := fmt.Sprintf(format, v...)
	l.buf = append(l.buf, s...)
	if len(s) == 0 || s[len(s)-1] != '\n' {
		l.buf = append(l.buf, '\n')
	}
	w.Write(l.buf)
}

func (l *Logger) logf(w io.Writer, level Level, prefix, format string, v ...interface{}) {
	if level >= l.level {
		l.output(w, prefix, format, v...)
	}
}

func (l *Logger) Debugf(format string, v ...interface{}) {
	l.logf(l.outWriter, Debug, "[D]", format, v...)
}

func (l *Logger) Infof(format string, v ...interface{}) {
	l.logf(l.outWriter, Info, "[I]", format, v...)
}

func (l *Logger) Warnf(format string, v ...interface{}) {
	l.logf(l.errWriter, Warn, "[W]", format, v...)
}

func (l *Logger) Errorf(format string, v ...interface{}) {
	l.logf(l.errWriter, Error, xterm.Warn.S("[E]"), format, v...)
}

func (l *Logger) Exitf(format string, v ...interface{}) {
	l.logf(l.errWriter, Error, xterm.Warn.S("[F]"), format, v...)
	os.Exit(1)
}

func (l *Logger) SetOutput(w io.Writer) {
	l.Lock()
	defer l.Unlock()
	l.outWriter = w
	l.errWriter = w
}

func (l *Logger) SetFlags(fs ...uint32) {
	var flags uint32
	for _, f := range fs {
		flags |= f
	}
	l.Lock()
	defer l.Unlock()
	l.flags = flags
}

var (
	Debugf    = std.Debugf
	Infof     = std.Infof
	Warnf     = std.Warnf
	Errorf    = std.Errorf
	Exitf     = std.Exitf
	SetFlags  = std.SetFlags
	SetOutput = std.SetOutput
)
