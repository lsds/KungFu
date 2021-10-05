import argparse
import time

from kungfu.python import current_cluster_size, current_rank
from kungfu.python.elastic_state import ElasticState, ElasticContext
from kungfu.python import propose_new_size
from kungfu.python import all_reduce_int_max


def parse_args():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--shuffle', action='store_true', default=False)
    p.add_argument('--max-step', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    rank = current_rank()
    size = current_cluster_size()
    print('%d/%d' % (rank, size))

    es = ElasticState(args.max_step, full_reload=True)
    progesss = es._progress

    i = 0
    while not es.stopped():
        with ElasticContext(es) as should_sync:
            if should_sync:
                print('should_sync')
                progesss = all_reduce_int_max(progesss)
                print('sync to progesss %d' % (progesss))
            i += 1
            print(i)
            progesss += 1
            time.sleep(0.01)

            if rank == 0:
                if i % 10 == 0:
                    new_size = i % 4 + 1
                    propose_new_size(new_size)
                    print('proposed %d' % (new_size))

    print('stopped, reasion: %s' % (es.stop_reason()))


main()
