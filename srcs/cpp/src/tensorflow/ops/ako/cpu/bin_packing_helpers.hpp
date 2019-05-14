#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <kungfu_tensorflow_ops.h>

#include <mutex>
#include <queue>

namespace tensorflow
{

    void bin_packing_simple(std::string input_tensor_name_,
                            Tensor* gradients, Tensor* output, 
                            DoneCallback done) {
        if (_partial_exchange_manager->isReadyForNegotiation(
                input_tensor_name_, global_step)) {
            _kungfu_world->AllReduce(gradients.tensor_data().data(),
                                     (void *)(output->tensor_data().data()),
                                     gradients.NumElements(),
                                     to_kungfu_type(gradients.dtype()),
                                     KungFu_SUM, name().c_str(), done);
            // Because it is synchronous, the done callback will signal when the
            // value held
            // in the memory where output points to is ready to be used.
        } else {
            *output = gradients;
            done();
        }
    }


     void bin_packing_accumulation(std::mutex allMutex, Tensor gradients, std::queue<Tensor> tensorWindow,
                                  bool isInit, bool hasInGrad, Tensor outGrad, Tensor inGrad, std::string input_tensor_name_,
                                  std::string average, Tensor* output, DoneCallback done) {
        std::lock_guard<std::mutex> lock(allMutex);

        // Update gradient window
        Tensor stale;
        tensorWindow.push(gradients);
        if (tensorWindow.size() >
            _partial_exchange_manager->partitions.size()) {
            stale = tensorWindow.front();
            tensorWindow.pop();
        }

        if (!isInit) {
            Tensor zeros(DataTypeToEnum<float>::v(), gradients.shape());
            auto zeros_flt = zeros.flat<float>();
            zeros_flt.setZero();
            outGrad = zeros;
            outGrad.flat<float>().setZero();
            inGrad = zeros;
            inGrad.flat<float>().setZero();
            isInit = true;
        }

        // Create snapshots right before you use the tensors
        if (hasInGrad) {
            gradients.flat<float>() =
                gradients.flat<float>() + inGrad.flat<float>();
            hasInGrad = false;
        }

        // Important: reset inbound gradients
        inGrad.flat<float>().setZero();

        // auto inGrad_flt  = inGrad.flat<float>();
        auto grads_flt   = gradients.flat<float>();
        auto outGrad_flt = outGrad.flat<float>();
        auto stale_flt   = stale.flat<float>();
        auto expire      = stale.NumElements() > 0;

        if (expire) {
            outGrad_flt = grads_flt + outGrad_flt - stale_flt;
        } else {
            outGrad_flt = grads_flt + outGrad_flt;
        }
        if (average_ == "peers") {
            // Divide by the number of peers
            // Similar issue encountered in Horovod:
            // https://github.com/horovod/horovod/issues/278 and
            // https://github.com/horovod/horovod/tree/fp16_divide_before_sum
            outGrad_flt = outGrad_flt / outGrad_flt.constant(num_peers_);
        } else if (average_ == "window") {
            if (tensorWindow.size() > 0) {
                outGrad_flt =
                    outGrad_flt / outGrad_flt.constant(tensorWindow.size());
            } else {
                std::cout << "Partial Exchange accumulation window empty!"
                          << std::endl;
            }
        }  // no average

        if (_partial_exchange_manager->isReadyForNegotiation(
                input_tensor_name_, global_step)) {
            // Create a callback to accumulate gradients from other peers
            std::function<void()> func = [&, done]() {
                std::lock_guard<std::mutex> l(allMutex);

                // subract gradients from inGrad to not apply them twice
                inGrad.flat<float>() =
                    inGrad.flat<float>() - gradients.flat<float>();
                hasInGrad = true;
                done();
            };

            _kungfu_world->AllReduce(
                outGrad.tensor_data().data(),
                (void *)(inGrad.tensor_data().data()), outGrad.NumElements(),
                to_kungfu_type(outGrad.dtype()), KungFu_SUM, name().c_str(),
                func);  // TODO: check deadlock here
            *output = gradients;
        } else {
            *output = gradients;
            done();
        }
    }


    void bin_packing_frontend_partitioning(int32_t global_step, std::string input_tensor_name_,
                            Tensor* gradients, Tensor* output, 
                            DoneCallback done, int32_t partitions_, int32_t index_) {
        if (global_step % partitions_ == index_) {
            _kungfu_world->AllReduce(gradients.tensor_data().data(),
                                     (void *)(output->tensor_data().data()),
                                     gradients.NumElements(),
                                     to_kungfu_type(gradients.dtype()),
                                     KungFu_SUM, name().c_str(), done);
            // Because it is synchronous, the done callback will signal when the
            // value held
            // in the memory where output points to is ready to be used.
        } else {
            *output = gradients;
            done();
        }
    }

}  // namespace tensorflow
