#include "tensorflow/contrib/lite/nnapi/internal/NeuralNetworksInterfaces.h"

namespace proto_nnapi {
namespace {

class NNPreparedModelImpl : public NNPreparedModel{
 public:
  NNPreparedModelImpl() = default;
  ~NNPreparedModelImpl() override {}
  ErrorStatus Execute(const Request& request,
                      std::function<void(ErrorStatus)> callback) {
    callback(ErrorStatus::NONE);
    return ErrorStatus::NONE;
  }
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
             std::shared_ptr<NNPreparedModel>(new NNPreparedModelImpl()));
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
