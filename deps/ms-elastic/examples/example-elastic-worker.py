import argparse
from kungfu.python.elastic_state import ElasticState, ElasticContext
from kungfu.python import current_rank, current_cluster_size, propose_new_size
from kungfu.python.elastic import create_tf_records


def parse_args():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--shuffle', action='store_true', default=False)
    p.add_argument('--reload', action='store_true', default=False)
    p.add_argument('--run', action='store_true', default=False)
    p.add_argument('--max-progress', type=int, default=10)
    p.add_argument('--global-batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--index-file', type=str, default='')
    return p.parse_args()


def ckpt(es):
    return 'progress-%010d.log' % (es._progress)


def read_step(es):
    with open(ckpt(es)) as f:
        return int(f.read().strip())


def save_step(es, step):
    with open(ckpt(es), 'w') as f:
        f.write('%d\n' % (step))


def main():
    args = parse_args()
    global_batch_size = args.global_batch_size

    es = ElasticState(args.max_progress, args.reload)
    progress = es._progress

    rank = current_rank()
    size = current_cluster_size()
    print('%d/%d, starting from %d' % (rank, size, progress))

    # TODO: sync dataset to progress

    if progress == 0 and rank == 0:
        propose_new_size(size)  # write identical config to config server

    if not args.run:
        return

    step = 0
    if rank == 0:
        if progress > 0:
            step = read_step(es)
            print('init step=%d' % (step))

    shard = create_tf_records(args.index_file, args.seed,
                              args.global_batch_size)

    filenames = shard['filenames']
    print('shard: %s' % (shard))
    print('%d files: %s' % (len(filenames), filenames))

    while not es.stopped():
        delta = global_batch_size
        with ElasticContext(es, delta) as should_sync:
            # print('# progress %d' % (es._progress))
            if should_sync:
                # TODO: In reload mode, NO need to sync dataset state
                # TODO: sync model states
                # FIXME: move it to es._progress
                pass

            step += 1
            progress = es._progress
            if rank == 0:
                # print('progress: %d/%s' % (progress, args.max_progress))
                pass

            # do step work
            #
            #

            if rank == 0:
                period = 300
                if step % period == 0:
                    new_size = (step // period) % 4 + 1
                    propose_new_size(new_size)

    progress = es._progress

    if rank == 0:
        save_step(es, step)


main()
