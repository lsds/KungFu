import json
import os


def _get_num_peers():
    cluster_spec = json.loads(os.getenv('KUNGFU_CLUSTER_SPEC'))
    return len(cluster_spec['Peers'])


def _get_self_rank():
    return int(os.getenv('KUNGFU_TEST_SELF_RANK'))


def _get_other_ranks():
    self_rank = _get_self_rank
    ranks = list(range(_get_num_peers()))
    return [r for r in ranks if r != self_rank]
