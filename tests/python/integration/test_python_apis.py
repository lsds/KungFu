# FIXME: make sure it runs without tensorflow
from kungfu.python import current_cluster_size, current_rank, run_barrier


def test_barrier():
    run_barrier()
    print(1)
    run_barrier()
    print(2)
    run_barrier()


def test_peer_info():
    rank = current_rank()
    np = current_cluster_size()
    print('rank=%d, np=%d' % (rank, np))


# TODO: more tests

test_barrier()
test_peer_info()
