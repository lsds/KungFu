// https://www.tensorflow.org/extend/adding_an_op
#pragma once

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow
{
// The AllReduce operator takes a single tensor (e.g. the computed gradient),
// and reduce (by taking sum) with the peers, and finally returns a tensor with
// exactly the same shape.
REGISTER_OP("AllReduce")
    .Attr("T: {int32, float32}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("Broadcast")
    .Attr("T: {int32, float32}")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("AkoNegotiator")
    .Input("allgradients: float32")
    .Input("partition: int32")
    .Input("partitioncount: int32")
    .Input("kickintime: int32")
    .Output("output: float32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("GlobalStepModifier")
    .Input("input: int32")
    .Output("output: int32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));  // TODO: don't require input
        // c->set_output(0, TensorShape());
        return Status::OK();
    });

REGISTER_OP("SetNumGradients").Input("input: int32");

REGISTER_OP("GlobalVariance").Input("input: float32");

}  // namespace tensorflow
