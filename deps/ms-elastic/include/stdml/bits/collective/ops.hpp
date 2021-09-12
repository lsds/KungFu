#pragma once
#include <stdml/bits/collective/session.hpp>
#include <stdml/tensor>

namespace ttl
{
template <typename R, typename D, typename S>
using basic_tensor_ref = internal::basic_tensor<R, D, S, internal::readwrite>;

template <typename R, typename D, typename S>
using basic_tensor_view = internal::basic_tensor<R, D, S, internal::readonly>;
}  // namespace ttl

namespace stdml::collective::ops
{
template <typename F>
class inplace_endo_op
{
  public:
    virtual void operator()(TensorRef y, TensorView x) const = 0;

    template <typename R, typename D, typename S>
    void operator()(ttl::basic_tensor_ref<R, D, S> x) const
    {
        (*this)(x, ttl::view(x));
    }

    void operator()(TensorRef x) const
    {
        (*this)(x, x);
    }
};

class basic_collective_op
{
  protected:
    session &sess_;

  public:
    basic_collective_op(session &sess) : sess_(sess)
    {
    }
};

class all_reduce : public basic_collective_op,
                   public inplace_endo_op<all_reduce>
{
    using P = basic_collective_op;
    using P::P;
    reduce_op op_;

  public:
    all_reduce(session &sess, reduce_op op = sum)
        : basic_collective_op(sess), op_(op)
    {
    }

    using inplace_endo_op<all_reduce>::operator();

    template <typename R, typename D, typename S>
    void operator()(ttl::basic_tensor_ref<R, D, S> y,
                    ttl::basic_tensor_view<R, D, S> x) const
    {
        sess_.all_reduce(x.data(), x.data_end(), y.data(), op_);
    }

    void operator()(TensorRef y, TensorView x) const
    {
        if (x.dtype() != stdml::f32) {
            fprintf(stderr, "all_reduce op only support f32\n");
            return;
        }
        using R = float;
        (*this)(y.typed<R>(), x.typed<R>());
    }

    uint32_t operator()(const uint32_t &x) const
    {
        return sess_.all_reduce(x, op_);
    }
};

class broadcast : public basic_collective_op, public inplace_endo_op<broadcast>
{
    using P = basic_collective_op;
    using P::P;

  public:
    using inplace_endo_op<broadcast>::operator();

    template <typename R, typename D, typename S>
    void operator()(ttl::basic_tensor_ref<R, D, S> y,
                    ttl::basic_tensor_view<R, D, S> x) const
    {
        sess_.broadcast(x.data(), x.data_end(), y.data());
    }

    template <typename R, typename D, typename S>
    void operator()(ttl::basic_tensor_ref<R, D, S> x) const
    {
        (*this)(x, ttl::view(x));
    }

    void operator()(TensorRef y, TensorView x) const
    {
        if (x.dtype() != stdml::f32) {
            fprintf(stderr, "broadcast op only support f32\n");
            return;
        }
        using R = float;
        (*this)(y.typed<R>(), x.typed<R>());
    }

    /*
/home/lg/opt/include/stdml/bits/collective/session.hpp:173:11: error: no
matching function for call to ‘stdml::Tensor::Tensor()’ T y;
    */
    // template <typename T>
    // T operator()(const T &x) const
    // {
    //     return sess_.broadcast(x);
    // }

    uint32_t operator()(const uint32_t &x) const
    {
        return sess_.broadcast(x);
    }
};
}  // namespace stdml::collective::ops
