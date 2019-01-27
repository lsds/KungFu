package shm

import "errors"

type ShmDarwin struct{ Shm } // TODO: implement

var errNotImplemented = errors.New("Not Imoplemented")

func New(name string) (Shm, error) {
	return &ShmDarwin{}, nil
}
