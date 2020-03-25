package utils

import "fmt"

func MergeErrors(errs []error, hint string) error {
	var msg string
	var failed int
	for _, e := range errs {
		if e != nil {
			failed++
			if len(msg) > 0 {
				msg += ", "
			}
			msg += e.Error()
		}
	}
	if failed == 0 {
		return nil
	}
	return fmt.Errorf("%s failed with %s: %s", hint, Pluralize(failed, "error", "errors"), msg)
}
