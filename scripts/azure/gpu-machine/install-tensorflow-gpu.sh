#!/bin/sh

set -e

sudo apt install -y python3 python3-pip

# Traceback (most recent call last):
#   File "/usr/bin/pip3", line 11, in <module>
#     sys.exit(main())
#   File "/usr/lib/python3/dist-packages/pip/__init__.py", line 215, in main
#     locale.setlocale(locale.LC_ALL, '')
#   File "/usr/lib/python3.5/locale.py", line 594, in setlocale
#     return _setlocale(category, locale)
# locale.Error: unsupported locale setting
export LC_ALL=C

pip3 install tensorflow-gpu
