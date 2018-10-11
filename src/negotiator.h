// https://www.tensorflow.org/extend/adding_an_op
#pragma once

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow
{
// A Negotiator takes a single tensor (the computed gradient), and negotiate (by
// taking mean, or delayed mean) with the peers, and finally returns a tensor
// with exactly the same shape.
REGISTER_OP("Negotiator")
    .Input("to_zero: float32")
    .Output("zeroed: float32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
}  // namespace tensorflow
