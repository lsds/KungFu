"""
Allow run the kungfu-run command via `python3.x.y -m kungfu.cmd.elastic_run ...`,
useful when multiple python versiosn are installed.

Usage:
    python3.x.y -m kungfu.cmd.elastic_run -np 4 python user-prog.py
"""
from kungfu.cmd import _elastic_run

if __name__ == '__main__':
    _elastic_run(2)  # skip `python3.x.y` and `-m`
