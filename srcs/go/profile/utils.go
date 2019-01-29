package profile

import (
	"fmt"
	"io"
	"unicode/utf8"
)

func padRight(n int, s string) string {
	for i := utf8.RuneCountInString(s); i < n; i++ {
		s = s + " "
	}
	return s
}

func showTable(w io.Writer, th []string, data [][]string) {
	var widths []int
	for _, title := range th {
		widths = append(widths, len(title))
	}
	for _, tr := range data {
		for i, td := range tr {
			if w := len(td); w > widths[i] {
				widths[i] = w
			}
		}
	}
	totalWidth := -5
	for _, w := range widths {
		totalWidth += w + 5
	}
	var hr string
	for i := 0; i < totalWidth; i++ {
		hr = hr + "-"
	}

	showRow := func(tr []string) {
		for i, td := range tr {
			if i > 0 {
				fmt.Fprintf(w, " ")
			}
			fmt.Fprintf(w, "%s", padRight(4+widths[i], td))
		}
		fmt.Fprintf(w, "\n")
	}
	showRow(th)
	fmt.Fprintf(w, "%s\n", hr)
	for _, tr := range data {
		showRow(tr)
	}
}
