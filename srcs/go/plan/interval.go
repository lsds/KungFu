package plan

// Interval represents the interval of integers [Begin, End)
type Interval struct {
	Begin int
	End   int
}

func (i Interval) Len() int { return i.End - i.Begin }

// EvenPartition parts an Interval into k parts such that the length of each part differ at most 1
func EvenPartition(r Interval, k int) []Interval {
	quo, rem := divide(r.Len(), k)
	var parts []Interval
	offset := r.Begin
	for i := 0; i < k; i++ {
		blockCount := func() int {
			if i < rem {
				return quo + 1
			}
			return quo
		}()
		parts = append(parts, Interval{Begin: offset, End: offset + blockCount})
		offset += blockCount
	}
	return parts
}

func divide(a, b int) (int, int) {
	q := a / b
	r := a - b*q
	return q, r
}
