package main

import (
	"flag"
	"os"
	"strconv"

	kf "github.com/lsds/KungFu/srcs/go/kungfu"
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/utils"
)

func main() {
	flag.Parse()
	algo := kb.ParseAlgo(os.Getenv(kb.AllReduceAlgoEnvKey))
	config := kf.Config{Algo: algo}
	kungfu, err := kf.New(config)
	if err != nil {
		utils.ExitErr(err)
	}

	kungfu.Start()
	defer kungfu.Close()
	work(kungfu)
	log.Infof("terminated")
}

func work(kungfu *kf.Kungfu) {
	initSession, err := strconv.Atoi(os.Getenv(kb.InitSessEnvKey))
	if err != nil {
		initSession = 0
	}

	needUpdate := false
	sessID := ""
	newSize := 1
	hostCap := 4

	for gs := initSession; gs <= hostCap; gs++ {
		if needUpdate {
			keep := kungfu.UpdateSession(sessID)
			if !keep {
				break
			}
		}
		train(kungfu)
		sessID = strconv.Itoa(gs + 1)
		newSize = gs + 1

		if gs < hostCap {
			kungfu.ProposeUpdate(gs+1, sessID, newSize)
			needUpdate = true
		}
	}
}

func train(kungfu *kf.Kungfu) {
	const dSize = 4
	dType := kb.KungFu_FLOAT
	size := 10
	sendBuf := &kb.Buffer{
		Data:  make([]byte, size*dSize),
		Count: size,
		Type:  dType,
	}
	recvBuf := &kb.Buffer{
		Data:  make([]byte, size*dSize),
		Count: size,
		Type:  dType,
	}
	w := kf.Workspace{
		SendBuf: sendBuf,
		RecvBuf: recvBuf,
		OP:      kb.KungFu_SUM,
		Name:    "!",
	}
	sess := kungfu.CurrentSession()
	sess.AllReduce(w)
}
