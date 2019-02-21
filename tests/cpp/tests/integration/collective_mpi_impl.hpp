#pragma once
#include <iostream>

#include <mpi.h>

template <typename T> struct mpi_type;
template <> struct mpi_type<int> {
    static auto value() { return MPI_INT; }
};
template <> struct mpi_type<float> {
    static auto value() { return MPI_FLOAT; }
};

struct mpi_collective {
    mpi_collective(int argc, char *argv[]) { MPI_Init(&argc, &argv); }
    ~mpi_collective() { MPI_Finalize(); }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char * /* FIXME: ignored */)
    {
        MPI_Allreduce(send_buf, recv_buf, count, mpi_type<T>::value(), MPI_SUM,
                      MPI_COMM_WORLD);
    }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char *name, std::function<void()> done)
    {
        // FIXME: not supported
        std::cerr << "mpi_collective::all_reduce<async> is not implemted"
                  << std::endl;
    }
};
