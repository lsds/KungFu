#!/bin/sh
set -e

mkdir -p $HOME/tmp && cd $HOME/tmp

MPI_MAJOR=3
MPI_MINOR=1
MPI_PATCH=1
VERSION=${MPI_MAJOR}.${MPI_MINOR}.${MPI_PATCH}

FILENAME=openmpi-${VERSION}.tar.bz2
FOLDER=openmpi-${VERSION}
URL=https://download.open-mpi.org/release/open-mpi/v${MPI_MAJOR}.${MPI_MINOR}/${FILENAME}

[ ! -f ${FILENAME} ] && curl -vLOJ $URL
tar -xf ${FILENAME}
cd ${FOLDER}

# will take about 8 min or longer depends on your machine
./configure
make -j $(nproc) all
sudo make install
sudo ldconfig
mpirun --version
