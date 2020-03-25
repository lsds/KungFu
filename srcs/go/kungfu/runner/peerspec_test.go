package runner

import "testing"

var tests = []struct {
	P PeerSpec
	S []string
}{
	{
		P: PeerSpec{
			Host: `x.y.z`,
			Port: 8081,
			Slot: 4,
		},
		S: []string{"x.y.z:8081:4"},
	},
	{
		P: PeerSpec{
			Host: `x.y.z`,
			Port: 8080,
		},
		S: []string{"x.y.z:8080:0", "x.y.z:8080"},
	},
}

func Test_PeerSpec(t *testing.T) {
	for _, tt := range tests {
		if s := tt.P.String(); s != tt.S[0] {
			t.Errorf("want %q, got %q", tt.S[0], s)
		}
		for _, s := range tt.S {
			p, err := ParsePeerSpec(s)
			if err != nil {
				t.Errorf("failed to parse %q: %v", s, err)
			}
			if *p != tt.P {
				t.Errorf("failed to parse %q, want: %v, got %v", s, tt.P, *p)
			}
		}
	}
}

func Test_PeerSpecList(t *testing.T) {
	var l PeerSpecList
	for _, tt := range tests {
		l = append(l, tt.P)
	}
	config := l.String()
	r, err := ParsePeerSpecList(config)
	if err != nil {
		t.Errorf("failed to parse %q: %v", config, err)
	}
	if !l.Eq(r) {
		t.Errorf("failed to parse %q", config)
	}

}

func Test_ParseEmptyPeerSpecList(t *testing.T) {
	l, err := ParsePeerSpecList("")
	if err != nil {
		t.Errorf("unexpect error: %v", err)
	}
	if n := len(l); n != 0 {
		t.Errorf("expect %d, got %d", 0, n)
	}
}

var validConfig = []string{
	`localhost:10000,localhost:10001,localhost:10008`,
}

func Test_ParsePeerSpecList(t *testing.T) {
	for _, tt := range validConfig {
		if _, err := ParsePeerSpecList(tt); err != nil {
			t.Errorf("failed to parse: %q", tt)
		}
	}
}
