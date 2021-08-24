"""
Allow run the kungfu-run command via `python3.x.y -m kungfu.cmd ...`,
useful when multiple python versiosn are installed.

Usage:
    python3.x.y -m kungfu.cmd -np 4 python user-prog.py
"""
from kungfu.cmd import run


if __name__ == '__main__':
    run(2)  # skip `python3.x.y` and `-m`
