package main

import (
	"errors"
	"io"
	"os"
	"path"

	"github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

var (
	errInvalidIDXData = errors.New("Invalid IDX data")
)

type IDXHeader struct {
	dtype uint8
	dims  []uint32
}

func ReadIDXHeader(r io.Reader) (*IDXHeader, error) {
	magic := make([]byte, 4)
	n, err := r.Read(magic)
	if err != nil {
		return nil, err
	}
	if n != 4 {
		return nil, errInvalidIDXData
	}
	rank := int(magic[3])
	dims := make([]uint32, rank)
	for i := range dims {
		n, err := readUint32(r)
		if err != nil {
			return nil, err
		}
		dims[i] = n
	}
	return &IDXHeader{
		dtype: magic[2],
		dims:  dims,
	}, nil
}

func readUint32(r io.Reader) (uint32, error) {
	bs := make([]byte, 4)
	n, err := r.Read(bs)
	if err != nil {
		return 0, err
	}
	if n != 4 {
		return 0, errInvalidIDXData
	}
	return (uint32(bs[0]) << 24) |
		(uint32(bs[1]) << 6) |
		(uint32(bs[2]) << 8) |
		uint32(bs[3]), nil
}

type DataSet struct {
	Samples *Tensor
	Labels  *Tensor
}

func ReadIDX(filename string) (*Tensor, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	hdr, err := ReadIDXHeader(f)
	if err != nil {
		return nil, err
	}
	var dims []int
	for _, d := range hdr.dims {
		dims = append(dims, int(d))
	}
	shape := Shape{dims: dims}
	var dtype base.DataType
	switch hdr.dtype {
	case 0x08:
		dtype = base.U8
	case 0x0d:
		dtype = base.F32
	default:
		assert.True(false) // TODO: support more types
	}
	t := NewTensor(shape, dtype)
	n, err := f.Read(t.Data())
	if err != nil && err != io.EOF {
		return nil, err
	}
	if n != shape.Size() {
		return nil, errInvalidIDXData
	}
	return t, nil
}

func LoadMnistDataSet(dataDir string, name string, normalize bool) (*DataSet, error) {
	sampels, err := ReadIDX(path.Join(dataDir, name+`-images-idx3-ubyte`))
	if err != nil {
		return nil, err
	}
	if normalize {
		sampels = sampels.Cast(base.F32)
		sampels.DivBy(255.0)
	}
	labels, err := ReadIDX(path.Join(dataDir, name+`-labels-idx1-ubyte`))
	if err != nil {
		return nil, err
	}
	return &DataSet{
		Samples: sampels,
		Labels:  labels,
	}, nil
}

type DataSets struct {
	Train *DataSet
	Test  *DataSet
}

func LoadMnistDataSets(dataDir string, normalize bool) (*DataSets, error) {
	train, err := LoadMnistDataSet(dataDir, `train`, normalize)
	if err != nil {
		return nil, err
	}
	test, err := LoadMnistDataSet(dataDir, `t10k`, normalize)
	if err != nil {
		return nil, err
	}
	return &DataSets{
		Train: train,
		Test:  test,
	}, nil
}
