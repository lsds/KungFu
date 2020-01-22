package main

import (
	"net/http"
	"os"
)

func main() {
	hostlistPath := os.Args[1]

	fileServer := http.FileServer(http.Dir(hostlistPath))
	http.Handle("/", http.StripPrefix("/", fileServer))

	http.ListenAndServe(":9100", nil)
}
