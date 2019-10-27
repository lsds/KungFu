package main

import (
	"flag"
	"fmt"
	"net/http"
)

var port = flag.Int("p", 9999, "")

func main() {
	http.ListenAndServe(fmt.Sprintf(":%d", *port), http.FileServer(http.Dir("./build/html")))
}
