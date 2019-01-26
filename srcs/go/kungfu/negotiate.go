package kungfu

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

func (sess *session) Warmup() int {
	k := sess.cluster.Size()
	count := k * 4
	dtype := kb.KungFu_INT32
	n := count * dtype.Size()
	w := Workspace{
		SendBuf: make([]byte, n),
		RecvBuf: make([]byte, n),
		Count:   count,
		Dtype:   dtype,
		OP:      kb.KungFu_SUM,
		Name:    "kungfu::warmup", // TODO: use tag
	}
	return code(sess.runStrategies(w, plan.EvenPartition, createCliqueStrategies(sess.cluster.Peers)))
}

func (sess *session) Negotiate(w Workspace) int {
	return code(sess.runStrategies(w, plan.EvenPartition, sess.strategies))
}

func code(err error) int {
	if err == nil {
		return 0
	}
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}
