package utils

func BytesEq(x, y []byte) bool {
	if len(x) != len(y) {
		return false
	}
	for i, a := range x {
		if a != y[i] {
			return false
		}
	}
	return true
}
