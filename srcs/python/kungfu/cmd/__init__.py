from kungfu.loader import _load_clib


def run():
    clib = _load_clib('libkungfu')
    clib.kungfu_run_main()
