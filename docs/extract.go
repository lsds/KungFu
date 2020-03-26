// +build ignore

package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"path"
	"strings"
)

var (
	kungfuRoot = flag.String("kungfu-root", ".", "path to kungfu git repo dir")
	pkg        = flag.String("pkg", "", "")
)

func main() {
	flag.Parse()
	mod, err := getGoMod(*kungfuRoot)
	if err != nil {
		panic(err)
	}
	dir := strings.TrimPrefix(*pkg, mod+"/")
	d := doc.New(parsePkg(dir), *pkg, doc.AllDecls)
	// fmt.Printf("import %q\n", *pkg)
	for _, t := range d.Types {
		if len(t.Doc) > 0 {
			fmt.Printf("* %s: %s\n", t.Name, t.Doc)
		}
	}
}

func parsePkg(dir string) *ast.Package {
	var fs token.FileSet
	pkgs, err := parser.ParseDir(&fs, dir, nil, parser.ParseComments)
	if err != nil {
		panic(err)
	}
	for _, p := range pkgs {
		// fmt.Printf("name: %s\n", k)
		return p
	}
	return nil
}

func getGoMod(dir string) (string, error) {
	f, err := os.Open(path.Join(dir, "go.mod"))
	if err != nil {
		return "", err
	}
	bs, err := ioutil.ReadAll(f)
	if err != nil {
		return "", err
	}
	lines := strings.Split(string(bs), "\n")
	parts := strings.Split(lines[0], " ")
	return parts[1], nil
}
