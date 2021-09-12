package main

import (
	"image"

	r "github.com/nfnt/resize"
)

func resize(img image.Image, h, w int) image.Image {
	return r.Resize(uint(w), uint(h), img, r.Lanczos3)
}
