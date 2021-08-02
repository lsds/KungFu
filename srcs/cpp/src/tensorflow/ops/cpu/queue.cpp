#include <kungfu/tensorflow/ops.h>
#include <kungfu/utils/ema.hpp>
#include <kungfu/utils/trace.hpp>

namespace tensorflow
{
REGISTER_KUNGFU_OP(NewQueue)  //
    .Attr("src: int")
    .Attr("dst: int")
    .Output("output: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    });

;

class NewQueue : public OpKernel
{
    using OpKernel::OpKernel;

    int src_;
    int dst_;

  public:
    explicit NewQueue(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("src", &src_));
        OP_REQUIRES_OK(context, context->GetAttr("dst", &dst_));
    }

    void Compute(OpKernelContext *context) override
    {
        int queueID = -1;
        kungfu::Peer::GetDefault()->NewQueue(src_, dst_, &queueID);

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, MakeTensorShape(), &output));
        output->scalar<int32_t>()() = queueID;
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(NewQueue, DEVICE_CPU);

REGISTER_KUNGFU_OP(QueuePut)  //
    .Attr("src: int")
    .Attr("dst: int")
    .Attr("qid: int")
    .Attr("T: {uint8, uint16, uint32, uint64, int8, int16, int32, int64, "
          "float16, float32, float64}")
    .Input("input: T")
    .SetIsStateful();

class QueuePut : public OpKernel
{
    using OpKernel::OpKernel;

    int src_;
    int dst_;
    int qid_;

  public:
    explicit QueuePut(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("src", &src_));
        OP_REQUIRES_OK(context, context->GetAttr("dst", &dst_));
        OP_REQUIRES_OK(context, context->GetAttr("qid", &qid_));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input = context->input(0);
        kungfu::Peer::GetDefault()->QueuePut(
            src_, dst_, qid_, input.tensor_data().data(), input.NumElements(),
            to_kungfu_type(input.dtype()));
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(QueuePut, DEVICE_CPU);

REGISTER_KUNGFU_OP(QueueGet)  //
    .Attr("T: {uint8, uint16, uint32, uint64, int8, int16, int32, int64, "
          "float16, float32, float64}")
    .Attr("src: int")
    .Attr("dst: int")
    .Attr("qid: int")
    .Attr("shape: shape")
    .Output("output: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        shape_inference::ShapeHandle handle;
        TensorShapeProto shape;
        TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape, &handle));
        c->set_output(0, handle);
        return Status::OK();
    });

class QueueGet : public OpKernel
{
    using OpKernel::OpKernel;

    TensorShapeProto shape_;

    int src_;
    int dst_;
    int qid_;

  public:
    explicit QueueGet(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("src", &src_));
        OP_REQUIRES_OK(context, context->GetAttr("dst", &dst_));
        OP_REQUIRES_OK(context, context->GetAttr("qid", &qid_));
        OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    }

    void Compute(OpKernelContext *context) override
    {
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape_, &output));
        kungfu::Peer::GetDefault()->QueueGet(
            src_, dst_, qid_, const_cast<char *>(output->tensor_data().data()),
            output->NumElements(), to_kungfu_type(output->dtype()));
    }
};

REGISTER_KUNGFU_KERNEL_BUILDER(QueueGet, DEVICE_CPU);
}  // namespace tensorflow
