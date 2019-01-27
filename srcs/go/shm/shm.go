package shm

type Shm interface {
	Close() error
	String() string
	Tell() int
	Seek(pos int) int
	Rewind()
	Write(bs []byte, n int) error
	Read(bs []byte, n int) error
}
