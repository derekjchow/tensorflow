#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_INTERFACES_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_INTERFACES_H

#include "tensorflow/contrib/lite/nnapi/internal/NeuralNetworksTypes.h"

#include <functional>
#include <memory>
#include <vector>

namespace proto_nnapi {

class NNPreparedModel {
 public:
  virtual ~NNPreparedModel() {}
  virtual ErrorStatus Execute(const Request& request,
                              std::function<void(ErrorStatus)> callback) = 0;
};

class NNDriver  {
 public:
  using PreparedModelCallback =
      std::function<void(ErrorStatus, std::shared_ptr<NNPreparedModel>)>;

  virtual ~NNDriver() {}
  virtual void GetCapabilities(
      std::function<void(ErrorStatus, const Capabilities&)> cb) = 0;
  virtual void GetSupportedOperations(
      const Model& model,
      std::function<void(ErrorStatus, const std::vector<bool>&)> cb) = 0;
  // Note(derekjchow): We don't distinguish between 1_0 models and 1_1 models
  // as they only differ by relax float32->float16 argument
  virtual ErrorStatus PrepareModel(
      const Model& model, PreparedModelCallback callback) = 0;
  virtual ErrorStatus PrepareModel_1_1(
      const Model& model,
      ExecutionPreference preference,
      PreparedModelCallback callback) = 0;
  virtual DeviceStatus GetStatus() = 0;
};

NNDriver* GetDriver();
}  // namespace proto_nnapi

#endif  // define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_INTERFACES_H
