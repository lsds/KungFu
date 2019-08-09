#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu/mst.hpp>
#include <kungfu_tensorflow_ops.h>

template <typename T> class has_member_t
{
    bool f(const T *begin, const T *end, const T &e) const
    {
        std::cerr << "looking up " << e << " in {";
        for (const T *p = begin; p < end; ++p) { std::cerr << *p << " "; }
        std::cerr << "}" << std::endl;
        for (const T *p = begin; p < end; ++p) {
            if (*p == e) { return true; }
        }
        return false;
    }

  public:
    bool operator()(const void *raw_begin, int n, const void *elem) const
    {
        const T *begin = reinterpret_cast<const T *>(raw_begin);
        return f(begin, begin + n, *reinterpret_cast<const T *>(elem));
    }
};

namespace tensorflow
{
bool has_member(const char *data, int n, const char *elem,
                const DataType &dtype)
{
    switch (dtype) {
    case DT_INT32:
        return has_member_t<int32_t>()(data, n, elem);
    case DT_INT64:
        return has_member_t<int64_t>()(data, n, elem);
    default:
        throw std::invalid_argument("unsupported dtype");
    }
}

REGISTER_OP("HasMember")
    .Attr("T: {int32, int64}")
    .Input("list: T")
    .Input("element: T")
    .Output("has_member: bool");

class HasMember : public OpKernel
{
    using OpKernel::OpKernel;

  public:
    void Compute(OpKernelContext *context) override
    {
        const auto list    = context->input(0);
        const auto element = context->input(1);
        Tensor *output     = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        output->scalar<bool>()() =
            has_member(list.tensor_data().data(), list.NumElements(),
                       element.tensor_data().data(), list.dtype());
        std::cout << "Result: " << output->scalar<bool>()() << std::endl;
    }
};

// TODO: use macro to add name prefix
REGISTER_KERNEL_BUILDER(Name("HasMember").Device(DEVICE_CPU), HasMember);

}  // namespace tensorflow
