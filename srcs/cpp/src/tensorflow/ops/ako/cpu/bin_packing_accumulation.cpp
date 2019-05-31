// #include <tensorflow/core/framework/op.h>
// #include <tensorflow/core/framework/op_kernel.h>
// #include <tensorflow/core/framework/shape_inference.h>

// #include <kungfu_tensorflow_ops.h>

// #include <mutex>
// #include <queue>

// namespace tensorflow
// {
// REGISTER_OP("PartialAccumulatingNegotiator")
//     .Attr("T: {int32, int64, float16, float32, float64}")
//     .Attr("input_tensor_name: string")
//     .Attr("average: string")
//     .Attr("budget: int")
//     .Attr("tensor_size: int")
//     .Attr("count_gradients: int")
//     .Attr("num_peers: int")
//     .Input("allgradients: T")
//     .Output("output: T")
//     .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
//         c->set_output(0, c->input(0));
//         return Status::OK();
//     });

// class PartialAccumulatingNegotiator : public AsyncOpKernel
// {
//     using AsyncOpKernel::AsyncOpKernel;
//     using CPUDevice = Eigen::ThreadPoolDevice;

//   public:
//     std::string input_tensor_name_;
//     int32_t tensorSize_;
//     int32_t count_gradients_;
//     int32_t budget;

//     int32_t num_peers_;
//     std::string average_;  // can be one of: peers, window, none

//     std::queue<Tensor> tensorWindow;
//     Tensor outGrad;  // the accumulated gradient to be negotiated
//     Tensor inGrad;   // the accumulated gradient received through negotiation
//     std::mutex allMutex;  // protects
//     bool hasInGrad;
//     bool isInit;

//     explicit PartialAccumulatingNegotiator(OpKernelConstruction *context)
//         : AsyncOpKernel(context)
//     {
//         OP_REQUIRES_OK(context, context->GetAttr("input_tensor_name",
//                                                  &input_tensor_name_));
//         OP_REQUIRES(
//             context, input_tensor_name_.size() >= 0,
//             errors::InvalidArgument("input_tensor_name must not be empty"));

//         OP_REQUIRES_OK(context, context->GetAttr("budget", &budget));
//         OP_REQUIRES(context, budget > 0,
//                     errors::InvalidArgument("budget must be greater than 0"));

//         OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &tensorSize_));
//         OP_REQUIRES(
//             context, tensorSize_ > 0,
//             errors::InvalidArgument("tensor size must be greater than 0"));

//         OP_REQUIRES_OK(context,
//                        context->GetAttr("count_gradients", &count_gradients_));
//         OP_REQUIRES(
//             context, count_gradients_ > 0,
//             errors::InvalidArgument("gradient count must be greater than 0"));

//         OP_REQUIRES_OK(context, context->GetAttr("num_peers", &num_peers_));
//         OP_REQUIRES(
//             context, num_peers_ > 0,
//             errors::InvalidArgument("peer count must be greater than 0"));

//         OP_REQUIRES_OK(context, context->GetAttr("average", &average_));

//         _partial_exchange_manager->setCountGradients(count_gradients_);
//         _partial_exchange_manager->setBudget(budget);
//         _partial_exchange_manager->addTensorInfo(input_tensor_name_,
//                                                  tensorSize_);

//         hasInGrad = false;
//         isInit    = false;
//     }

//     void ComputeAsync(OpKernelContext *context, DoneCallback done) override
//     {
//         DCHECK_EQ(1, context->num_inputs());

//         Tensor &gradients = (Tensor &)context->input(0);

//         Tensor *output = nullptr;
//         OP_REQUIRES_OK(context,
//                        context->allocate_output(0, gradients.shape(), &output));

//         std::lock_guard<std::mutex> lock(allMutex);

//         // Update gradient window
//         Tensor stale;
//         tensorWindow.push(gradients);
//         if (tensorWindow.size() >
//             _partial_exchange_manager->partitions.size()) {
//             stale = tensorWindow.front();
//             tensorWindow.pop();
//         }

//         if (!isInit) {
//             Tensor zeros(DataTypeToEnum<float>::v(), gradients.shape());
//             auto zeros_flt = zeros.flat<float>();
//             zeros_flt.setZero();
//             outGrad = zeros;
//             outGrad.flat<float>().setZero();
//             inGrad = zeros;
//             inGrad.flat<float>().setZero();
//             isInit = true;
//         }

//         // Create snapshots right before you use the tensors
//         if (hasInGrad) {
//             gradients.flat<float>() =
//                 gradients.flat<float>() + inGrad.flat<float>();
//             hasInGrad = false;
//         }

//         // Important: reset inbound gradients
//         inGrad.flat<float>().setZero();

//         // auto inGrad_flt  = inGrad.flat<float>();
//         auto grads_flt   = gradients.flat<float>();
//         auto outGrad_flt = outGrad.flat<float>();
//         auto stale_flt   = stale.flat<float>();
//         auto expire      = stale.NumElements() > 0;

//         if (expire) {
//             outGrad_flt = grads_flt + outGrad_flt - stale_flt;
//         } else {
//             outGrad_flt = grads_flt + outGrad_flt;
//         }
//         if (average_ == "peers") {
//             // Divide by the number of peers
//             // Similar issue encountered in Horovod:
//             // https://github.com/horovod/horovod/issues/278 and
//             // https://github.com/horovod/horovod/tree/fp16_divide_before_sum
//             outGrad_flt = outGrad_flt / outGrad_flt.constant(num_peers_);
//         } else if (average_ == "window") {
//             if (tensorWindow.size() > 0) {
//                 outGrad_flt =
//                     outGrad_flt / outGrad_flt.constant(tensorWindow.size());
//             } else {
//                 std::cout << "Partial Exchange accumulation window empty!"
//                           << std::endl;
//             }
//         }  // no average

//         if (_partial_exchange_manager->isReadyForNegotiation(
//                 input_tensor_name_, _kungfu_world->GetGlobalStep())) {
//             // Create a callback to accumulate gradients from other peers
//             std::function<void()> func = [&, done]() {
//                 std::lock_guard<std::mutex> l(allMutex);

//                 // subract gradients from inGrad to not apply them twice
//                 inGrad.flat<float>() =
//                     inGrad.flat<float>() - gradients.flat<float>();
//                 hasInGrad = true;
//                 done();
//             };

//             _kungfu_world->AllReduce(
//                 outGrad.tensor_data().data(),
//                 (void *)(inGrad.tensor_data().data()), outGrad.NumElements(),
//                 to_kungfu_type(outGrad.dtype()), KungFu_SUM, name().c_str(),
//                 func);  // TODO: check deadlock here
//             *output = gradients;
//         } else {
//             *output = gradients;
//             done();
//         }
//     }
// };

// REGISTER_KERNEL_BUILDER(
//     Name("PartialAccumulatingNegotiator").Device(DEVICE_CPU),
//     PartialAccumulatingNegotiator);

// }  // namespace tensorflow
