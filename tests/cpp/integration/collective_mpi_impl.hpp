#pragma once
#include <functional>
#include <iostream>

#include <mpi.h>

#include <kungfu/utils/error_checker.hpp>

struct show_mpi_error {
    std::string operator()(int code) const
    {
        return "mpi_err_code: " + std::to_string(code);
    }
};

using mpi_checker = error_checker<int, MPI_SUCCESS, show_mpi_error>;

template <typename T> struct mpi_type;
template <> struct mpi_type<uint8_t> {
    static auto value() { return MPI_INT8_T; }
};
template <> struct mpi_type<int> {
    static auto value() { return MPI_INT; }
};
template <> struct mpi_type<float> {
    static auto value() { return MPI_FLOAT; }
};

class mpi_collective
{
    const int _root;
    int _rank;
    int _cluster_size;

  public:
    mpi_collective(int argc, char *argv[]) : _root(0)
    {
        KUNGFU_CHECK(mpi_checker) << MPI_Init(&argc, &argv);
        KUNGFU_CHECK(mpi_checker) << MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        KUNGFU_CHECK(mpi_checker)
            << MPI_Comm_size(MPI_COMM_WORLD, &_cluster_size);
        printf("MPI inited: %d/%d\n", _rank, _cluster_size);
    }

    ~mpi_collective()
    {
        printf("before MPI_Finalize: %d/%d\n", _rank, _cluster_size);
        KUNGFU_CHECK(mpi_checker) << MPI_Finalize();
        printf("MPI finalized: %d/%d\n", _rank, _cluster_size);
    }

    bool is_root() const { return _root == _rank; }

    int rank() const { return _rank; }

    int cluster_size() const { return _cluster_size; }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char * /* FIXME: ignored */)
    {
        KUNGFU_CHECK(mpi_checker)
            << MPI_Allreduce(send_buf, recv_buf, count, mpi_type<T>::value(),
                             MPI_SUM, MPI_COMM_WORLD);
    }

    template <typename T>
    void all_reduce(const T *send_buf, T *recv_buf, size_t count,
                    const char *name, std::function<void()> done)
    {
        // FIXME: not supported
        std::cerr << "mpi_collective::all_reduce<async> is not implemted"
                  << std::endl;
        done();
    }

    template <typename T>
    void bcast(T *buf, size_t count, const char * /* name */)
    {
        KUNGFU_CHECK(mpi_checker) << MPI_Bcast(buf, count, mpi_type<T>::value(),
                                               _root, MPI_COMM_WORLD);
    }
};
