package kungfu

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

func (kf *Kungfu) Warmup() {
	k := kf.currentCluster().Size()
	count := k * 4
	dtype := kb.KungFu_INT32
	n := count * dtype.Size()
	sendBuf := make([]byte, n)
	recvBuf := make([]byte, n)
	op := kb.KungFu_SUM
	name := "kungfu::warmup" // TODO: use tag
	kf.runPartitions(sendBuf, recvBuf, count, dtype, op, name, plan.EvenPartition, createCliquePartitions(kf.currentCluster().Peers))
}

func (kf *Kungfu) Negotiate(sendBuf []byte, recvBuf []byte, count int, dtype kb.KungFu_Datatype, op kb.KungFu_Op, name string) int {
	return code(kf.runPartitions(sendBuf, recvBuf, count, dtype, op, name, plan.EvenPartition, kf.initSession.partitions))
}

func code(err error) int {
	if err == nil {
		return 0
	}
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}
