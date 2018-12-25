package main

func ceilDiv(a, b int) int {
	q := a / b
	if a%b != 0 {
		q++
	}
	return q
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
