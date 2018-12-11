package main

import "testing"

func Test_replaceIfExists(t *testing.T) {
	newValues := map[string]string{
		`X`: `2`,
	}

	want := `X=2`
	if got := replaceIfExists(`X=`, newValues); got != want {
		t.Errorf("want: %s, got %s", want, got)
	}
}
