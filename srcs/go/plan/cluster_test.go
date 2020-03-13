package plan

import "testing"

func Test_Resize(t *testing.T) {
	r1 := PeerID{IPv4: 1, Port: 31300}
	r2 := PeerID{IPv4: 2, Port: 31200}

	w1 := PeerID{IPv4: 1, Port: 100}
	w2 := PeerID{IPv4: 1, Port: 101}
	w3 := PeerID{IPv4: 2, Port: 100}
	c := Cluster{
		Runners: PeerList{r1, r2},
		Workers: PeerList{w1, w2, w3},
	}
	w4 := PeerID{IPv4: 2, Port: 101}

	d, err := c.Resize(4)
	if err != nil || !(d.Workers[3] == w4) {
		t.Errorf("invalid resize")
	}
}
