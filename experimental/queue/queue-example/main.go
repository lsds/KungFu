package main

import (
	"flag"
	"fmt"

	"github.com/lsds/KungFu/srcs/go/kungfu/peer"
	"github.com/lsds/KungFu/srcs/go/kungfu/session"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func main() {
	flag.Parse()
	peer, err := peer.New()
	if err != nil {
		utils.ExitErr(err)
	}
	peer.Start()
	defer peer.Close()

	sess := peer.CurrentSession()
	log.Infof("%d/%d", sess.Rank(), sess.Size())

	sess.Barrier()
	f(sess)
	sess.Barrier()
}

func f(sess *session.Session) {
	ranks := []int{1, 2}
	s := make(map[int]struct{})
	for _, r := range ranks {
		s[r] = struct{}{}
	}

	if _, ok := s[sess.Rank()]; ok {
		q, err := sess.NewQueue(1, 2)
		if err != nil {
			return
		}
		if sess.Rank() == 1 {
			// x := base.NewVector(1, base.I32)
			for i := 0; i < 10; i++ {
				x := []byte(`12345`)
				if err := q.Put(x); err != nil {
					utils.ExitErr(err)
				}
			}
		} else {
			for i := 0; i < 10; i++ {

				y, err := q.Get()
				if err != nil {
					utils.ExitErr(err)
				}
				fmt.Printf("received: %s\n", y)
			}
		}
		//
	} else {
		//
	}
}
