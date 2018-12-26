package main

func divide(a, b int) (int, int) {
	q := a / b
	r := a - b*q
	return q, r
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
