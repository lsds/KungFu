package kungfu

import (
	"github.com/luomai/kungfu/srcs/go/log"
	"github.com/luomai/kungfu/srcs/go/wire"
)

func (kf *Kungfu) Negotiate(sendBuf []byte, recvBuf []byte, count int, dtype wire.KungFu_Datatype, op wire.KungFu_Op, name string) int {
	k := kf.cluster.Size()
	switch kf.config.Algo {
	case wire.KungFu_Tree:
		return code(kf.treeAllReduce(sendBuf, recvBuf, count, dtype, op, name))
	case wire.KungFu_Clique:
		if count >= k {
			return code(kf.fullSymmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name))
		}
		log.Warnf("data size (%d) is smaller that cluster size, will not use fullSymmetricAllReduce", count, k)
	case wire.KungFu_Ring:
		if count >= k && k >= 3 {
			return code(kf.ringAllReduce(sendBuf, recvBuf, count, dtype, op, name))
		}
	case wire.KungFu_Simple:
		return code(kf.simpleAllReduce(sendBuf, recvBuf, count, dtype, op, kf.cluster.Root(), name))
	}

	log.Warnf("fallback to simpleAllReduce")
	return code(kf.simpleAllReduce(sendBuf, recvBuf, count, dtype, op, kf.cluster.Root(), name))
}

func code(err error) int {
	if err == nil {
		return 0
	}
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}
