package plan

import "encoding/json"

func toString(i interface{}) string {
	bs, err := json.Marshal(i)
	if err != nil {
		return ""
	}
	return string(bs)
}

func FromString(s string, i interface{}) error {
	return json.Unmarshal([]byte(s), i)
}
