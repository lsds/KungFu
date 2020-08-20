// +build ignore

package main

import (
	"flag"
	"fmt"
	"net/http"
)

var port = flag.Int("port", 9999, "")

func main() {
	flag.Parse()
	http.ListenAndServe(fmt.Sprintf(":%d", *port), http.FileServer(http.Dir("./build/html")))
}
