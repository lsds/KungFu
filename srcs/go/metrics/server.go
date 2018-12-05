package metrics

import (
	"encoding/json"
	"expvar"
	"io"
	"net"
	"net/http"
	"strconv"
	"time"
)

var (
	StartTime = expvar.NewInt("start_timestamp_ns")
	StopTime  = expvar.NewInt("stop_timestamp_ns")
)

func init() {
	StartTime.Set(time.Now().UnixNano())
	StopTime.Set(0)
}

func RecordStop() {
	StopTime.Set(time.Now().UnixNano())
}

func ListenAndServe(port uint16) {
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(int(port)))
	http.ListenAndServe(addr, expvar.Handler())
}

func Export(w io.Writer) {
	m := make(map[string]interface{})
	expvar.Do(func(kv expvar.KeyValue) {
		var i interface{}
		if err := json.Unmarshal([]byte(kv.Value.String()), &i); err == nil {
			m[kv.Key] = i
		}
	})
	e := json.NewEncoder(w)
	e.SetIndent("", "    ")
	e.Encode(m)
}
