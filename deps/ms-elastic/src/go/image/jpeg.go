package main

import "C"
import (
	"bytes"
	"fmt"
	"image"
	"log"
	"reflect"
	"unsafe"

	_ "image/jpeg"
)

func decodeJpeg(bs []byte) {
	r := bytes.NewBuffer(bs)
	config, format, err := image.DecodeConfig(r)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Width:", config.Width, "Height:", config.Height, "Format:", format)

}

func decodeJpegHW3(bs []byte, h, w int, bmp []byte) int {
	r := bytes.NewBuffer(bs)
	img, _, err := image.Decode(r)
	if err != nil {
		return 1
	}
	img = resize(img, h, w)
	var idx int
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			color := img.At(j, i) // TODO: verify!
			r, g, b, _ := color.RGBA()
			bmp[idx+2] = uint8(r)
			bmp[idx+1] = uint8(g)
			bmp[idx+0] = uint8(b)
			idx += 3
		}
	}
	return 0
}

//export GoDecodeJpeg
func GoDecodeJpeg(ptr unsafe.Pointer, ptrSize int) {
	decodeJpeg(goBytes(ptr, ptrSize))
}

//export GoDecodeJpegHW3
func GoDecodeJpegHW3(ptr unsafe.Pointer, ptrSize int, h, w int, bmp unsafe.Pointer) int {
	return decodeJpegHW3(goBytes(ptr, ptrSize), h, w, goBytes(bmp, h*w*3))
}

func main() {}

func goBytes(ptr unsafe.Pointer, ptrSize int) []byte {
	sh := &reflect.SliceHeader{
		Data: uintptr(ptr),
		Len:  ptrSize,
		Cap:  ptrSize,
	}
	return *(*[]byte)(unsafe.Pointer(sh))
}
