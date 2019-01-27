package shm

/*
#cgo LDFLAGS: -lrt

#include <stdlib.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Shm wraps a POSIX shared memory objects, see man `man shm_open` for more details.
type ShmLinux struct {
	name *C.char
	fd   C.int
}

var (
	ErrShmOpenFailed   = errors.New("shm_open failed")
	ErrShmUnlinkFailed = errors.New("shm_unlink failed")
)

// New creates a Shm with shm_open.
func New(name string) (Shm, error) {
	cname := C.CString(name)
	fd := C.shm_open(cname, C.O_RDWR|C.O_CREAT, 0777)
	if fd <= 0 {
		return nil, ErrShmOpenFailed
	}
	return &ShmLinux{name: cname, fd: fd}, nil
}

// Close deletes a Shm with shm_unlink.
func (s *ShmLinux) Close() error {
	defer C.free(unsafe.Pointer(s.name))
	if code := C.shm_unlink(s.name); code != 0 {
		return ErrShmUnlinkFailed
	}
	return nil
}

func (s *ShmLinux) String() string {
	return fmt.Sprintf("shm<%s,fd=%d,offset=%d>", C.GoString(s.name), s.fd, s.Tell())
}

func (s *ShmLinux) Tell() int {
	return int(C.lseek(s.fd, 0, C.SEEK_CUR))
}

func (s *ShmLinux) Seek(pos int) int {
	return int(C.lseek(s.fd, C.long(pos), C.SEEK_SET))
}

func (s *ShmLinux) Rewind() {
	s.Seek(0)
}

func (s *ShmLinux) Write(bs []byte, n int) error {
	if wrote := int(C.write(s.fd, ptr(bs), C.ulong(n))); wrote != n {
		return fmt.Errorf("want %d, wrote %d", n, wrote)
	}
	return nil
}

func (s *ShmLinux) Read(bs []byte, n int) error {
	if got := int(C.read(s.fd, ptr(bs), C.ulong(n))); got != n {
		return fmt.Errorf("want %d, got %d", n, got)
	}
	return nil
}

func ptr(bs []byte) unsafe.Pointer {
	return unsafe.Pointer(&bs[0])
}
