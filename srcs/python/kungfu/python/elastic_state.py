from kungfu.python import current_rank, current_cluster_size, resize, all_reduce_int_max, change_cluster, init_progress


class ElasticState:
    def __init__(self, max_progress=None, full_reload=False):
        self._progress = init_progress()
        self._max_progress = max_progress
        self._synced = False
        self._stop_reason = None
        # print('full_reload=%s' % (full_reload))
        self._full_reload = full_reload

    def begin(self):
        should_sync = not self._synced
        if should_sync:
            if self._full_reload:
                # FIXME: no need to sync
                new_progress = all_reduce_int_max(self._progress)
                if new_progress != self._progress:
                    raise RuntimeError(
                        "invalid init_progress in full reload mode detected")
            else:
                # print('sync new_progress')
                new_progress = all_reduce_int_max(self._progress)
                # print('synced new_progress')
            self._progress = new_progress
            self._synced = True
        return should_sync

    def _resize(self):
        if self._full_reload:
            # print('calling new API')
            return change_cluster(self._progress)
        else:
            # print('calling old API')
            return resize()

    def end(self, delta_progress=1):
        self._progress += delta_progress
        if self._max_progress:
            if self._progress >= self._max_progress:
                self._stop_reason = 'finished'
                return

        # print('checking resize')
        changed, detached = self._resize()
        if changed:
            # print('changed')
            if detached:
                self._stop_reason = 'detached'
                return
            elif self._full_reload:
                # print('elastic mode is reload')
                self._stop_reason = 'reload'
                return

            self._synced = False
        else:
            pass
            # print('not changed')

    def stopped(self):
        return self._stop_reason is not None

    def stop_reason(self):
        return self._stop_reason


class ElasticContext:
    def __init__(self, elastic_state, delta_progress=1):
        self._elastic_state = elastic_state
        self._delta_progress = delta_progress

    def __enter__(self):
        return self._elastic_state.begin()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._elastic_state.end(self._delta_progress)
