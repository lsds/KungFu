package kungfu

import (
	"sync/atomic"

	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/log"
)

func (kf *Kungfu) Warmup() {
	k := kf.cluster.Size()
	count := k * 4
	dtype := kb.KungFu_INT32
	n := count * dtype.Size()
	sendBuf := make([]byte, n)
	recvBuf := make([]byte, n)
	op := kb.KungFu_SUM
	name := "kungfu::warmup" // TODO: use tag
	kf.fullSymmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name)
}

func (kf *Kungfu) Negotiate(sendBuf []byte, recvBuf []byte, count int, dtype kb.KungFu_Datatype, op kb.KungFu_Op, name string) int {
	k := kf.cluster.Size()
	switch kf.config.Algo {
	case kb.KungFu_Tree:
		return code(kf.treeAllReduce(sendBuf, recvBuf, count, dtype, op, name))
	case kb.KungFu_Clique:
		if count >= k {
			return code(kf.fullSymmetricAllReduce(sendBuf, recvBuf, count, dtype, op, name))
		}
		infrequently.Do(func() {
			log.Warnf("data size (%d) is smaller that cluster size %d, will not use fullSymmetricAllReduce", count, k)
		})
	case kb.KungFu_Ring:
		if count >= k && k >= 3 {
			return code(kf.ringAllReduce(sendBuf, recvBuf, count, dtype, op, name))
		}
	case kb.KungFu_Simple:
		return code(kf.simpleAllReduce(sendBuf, recvBuf, count, dtype, op, kf.cluster.Root(), name))
	}
	infrequently.Do(func() { log.Warnf("%s is not implemeted, fallback to simpleAllReduce", kf.config.Algo) })
	return code(kf.simpleAllReduce(sendBuf, recvBuf, count, dtype, op, kf.cluster.Root(), name))
}

func code(err error) int {
	if err == nil {
		return 0
	}
	// TODO: https://www.open-mpi.org/doc/v3.1/man3/MPI.3.php#sect4
	return 1
}

var infrequently Infrequently

type Infrequently struct {
	period               int64
	done                 int64
	skippedSinceLastDone int64
}

func (i *Infrequently) Do(f func()) {
	c := atomic.LoadInt64(&i.skippedSinceLastDone)
	p := atomic.LoadInt64(&i.period)
	if c == p {
		f()
		d := atomic.AddInt64(&i.done, 1)
		atomic.StoreInt64(&i.skippedSinceLastDone, 0)
		if d >= p {
			atomic.StoreInt64(&i.period, d*50)
		}
		return
	}
	atomic.AddInt64(&i.skippedSinceLastDone, 1)
}
