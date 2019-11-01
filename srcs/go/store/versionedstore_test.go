package store

import (
	"testing"
)

func Test_1(t *testing.T) {
	vs := NewVersionedStore(1)

	a, err := vs.Create("0xff", "a.idx", 1)
	a.Data[0] = 1
	if _, err := vs.Create("0xff", "a.idx", 1); err == nil {
		t.Error("Create should return error")
	}

	if _, err := vs.Get("0x00", "a.idx"); err == nil {
		t.Error("Get should return error")
	}
	b, err := vs.Get("0xff", "a.idx")
	if err != nil {
		t.Error("unexpected error")
	}

	if b.Data[0] != 1 {
		t.Error("Get failed")
	}

	vs.Create("0x100", "a.idx", 1)
	if _, err := vs.Get("0xff", "a.idx"); err == nil {
		t.Error("Get should return error")
	}
}
