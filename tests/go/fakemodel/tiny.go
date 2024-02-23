package fakemodel

func genFakeModel(k, n int) []int {
	a := make([]int, n)
	for i := range a {
		a[i] = k
	}
	return a
}
