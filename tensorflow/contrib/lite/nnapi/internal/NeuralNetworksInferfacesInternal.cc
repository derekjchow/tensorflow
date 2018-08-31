#include "tensorflow/contrib/lite/nnapi/internal/NeuralNetworksInterfaces.h"

#include <iostream>

#define LOG(x) std::cerr

namespace proto_nnapi {
namespace {

class NNPreparedModelImpl : public NNPreparedModel {
 public:
  NNPreparedModelImpl(const Model& model)
   : model_(model) {}

  ~NNPreparedModelImpl() override {}

  ErrorStatus Execute(const Request& request,
                      std::function<void(ErrorStatus)> callback) {
    LOG(ERROR) << "\n----Running Model----\n";
    LOG(ERROR) << "\n----Model stats----\n";
    LOG(ERROR) << "Operands: " << model_.operands().size() << "\n";
    LOG(ERROR) << "Operations: " << model_.operations().size() << "\n";
    LOG(ERROR) << "Inputs: " << model_.input_indexes().size() << "\n";
    LOG(ERROR) << "Outputs: " << model_.output_indexes().size() << "\n";
    LOG(ERROR) << "Small value size: " << model_.small_values().size() << "\n";
    LOG(ERROR) << "Pools: " << model_.NumPools() << "\n";
    LOG(ERROR) << "----Model stats----" << "\n\n\n";

    const std::vector<OperandType>& operands = model_->operands();

    for (const Operation& operation : model_.operations()) {
      const auto& inputs = operation.inputs;
      const auto& outputs = operation.outputs;
      // TODO(derekjchow): Log input types and such.
      switch (operation.type) {
        case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
          LOG(ERROR) << "Running DEPTHWISE_CONV_2D op\n";
          break;
        case ANEURALNETWORKS_CONV_2D:
          LOG(ERROR) << "Running CONV2D op\n";
          if (operation.inputs.size()) {
            LOG(ERROR) << "Unsupported 10 input conv\n";
            break;
          }
          int32_t padding_implicit = operands[inputs[3]];
          stride_width     = operands[inputs[4]];
          stride_height    = operands[inputs[5]];
          activation       = operands[inputs[6]];
          break;
        default:
          LOG(ERROR) << "Running operation of type: " << operation.type << "\n";
      }
    }

    LOG(ERROR) << "\n\n----Executed Model----\n\n";

    callback(ErrorStatus::NONE);
    return ErrorStatus::NONE;
  }

 private:

  const Model model_;
};

class NNDriverImpl : public NNDriver {
 public:

  NNDriverImpl() = default;
  ~NNDriverImpl() override {}

  // These do nothing at the moment. TODO(derekjchow): Make version that do.
  void GetCapabilities(
      std::function<void(ErrorStatus, const Capabilities&)> cb) override {}

  void GetSupportedOperations(
      const Model& model,
      std::function<void(ErrorStatus, const std::vector<bool>&)> cb) override {}

  // Note(derekjchow): We don't distinguish between 1_0 models and 1_1 models
  // as they only differ by relax float32->float16 argument
  ErrorStatus PrepareModel(const Model& model,
                           PreparedModelCallback callback) override {
    callback(ErrorStatus::NONE,
             std::shared_ptr<NNPreparedModel>(new NNPreparedModelImpl(model)));
    return ErrorStatus::NONE;
  }

  ErrorStatus PrepareModel_1_1(const Model& model,
                               ExecutionPreference preference,
                               PreparedModelCallback callback) override {
    return PrepareModel(model, callback);
  }

  DeviceStatus GetStatus() override {
    return DeviceStatus::AVAILABLE;
  }
};

}  // namespace

NNDriver* GetDriver() {
  static std::unique_ptr<NNDriverImpl> g_driver;
  if (!g_driver) {
    g_driver.reset(new NNDriverImpl());
  }
  return g_driver.get();
}

}  // namespace proto_nnapi
