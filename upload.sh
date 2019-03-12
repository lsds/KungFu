#!/bin/sh
set -e

if [ -d 3rdparty ]; then rm -fr 3rdparty; fi
git clean -fdx

remote_dir=dbg-multi-machine

ssh -v platypus2.doc.res.ic.ac.uk mkdir -p $remote_dir

time rsync -v -r . platypus2.doc.res.ic.ac.uk:~/$remote_dir/

ssh platypus2.doc.res.ic.ac.uk ./$remote_dir/x.sh
