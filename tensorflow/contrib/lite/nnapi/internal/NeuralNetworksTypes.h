#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_TYPES_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_TYPES_H

#include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"

#include <vector>


// This is copied straight from NN API
enum class ErrorStatus : int32_t {
    NONE = 0,
    DEVICE_UNAVAILABLE = 1,
    GENERAL_FAILURE = 2,
    OUTPUT_INSUFFICIENT_SIZE = 3,
    INVALID_ARGUMENT = 4,
};

// Also copied straight from NN API
enum class ExecutionPreference : int32_t {
    LOW_POWER = 0,
    FAST_SINGLE_ANSWER = 1,
    SUSTAINED_SPEED = 2,
};

// Also copied straight from NN API
enum class DeviceStatus : int32_t {
    AVAILABLE = 0,
    BUSY = 1,
    OFFLINE = 2,
    UNKNOWN = 3,
};

struct Memory {
  size_t size;
  int prot;
  int fd;
  size_t offset;
};

struct DataLocation {
  uint32_t pool_index;
  uint32_t offset;
  uint32_t length;
};

struct RequestArgument {
  bool has_no_value;
  DataLocation location;
  std::vector<uint32_t> dimensions;
};

struct Request {
  RequestArgument inputs;
  RequestArgument outputs;

  // TODO(derekjchow): Figure out what are these pools....
  std::vector<Memory> pools;
};

// TODO(derekjchow): Flesh these out
struct Capabilities {
  // neuralnetworks::V1_0::PerformanceInfo float32Performance;
  // neuralnetworks::V1_0::PerformanceInfo quantized8Performance;
};

enum class OperandLifeTime {
  TEMPORARY_VARIABLE,
  MODEL_INPUT,
  MODEL_OUTPUT,
  CONSTANT_COPY,
  CONSTANT_REFERENCE,
  NO_VALUE,
};

struct OperandType {
  int32_t type;
  std::vector<uint32_t> dimensions;
  float scale = 0.0f;
  int32_t zero_point = 0;

  OperandLifeTime lifetime = OperandLifeTime::TEMPORARY_VARIABLE;
  DataLocation location = {.pool_index = 0, .offset = 0, .length = 0};
};

// This is a HIDL struct
struct Operation {
  ANeuralNetworksOperationType type;
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;
};

// Large values are used as placeholders for operand's where shared memory
// hasn't been allocated yet. These values will get allocated later.
struct LargeValue {
  uint32_t operand_index;
  const void* buffer;
};

class Model {
 public:
  Model();
  ~Model();

  int Finish();

  int AddOperand(const OperandType& type);

  int SetOperandValue(uint32_t index, const void* buffer, size_t length);

  int SetOperandValueFromMemory(uint32_t index, const Memory* memory,
                                uint32_t offset, size_t length);

  void AddOperation(ANeuralNetworksOperationType type,
                    std::vector<uint32_t> inputs,
                    std::vector<uint32_t> outputs);

  void IdentifyInputsAndOutputs(std::vector<uint32_t> inputs,
                                std::vector<uint32_t> outputs);

  //void RelaxComputationFloat32toFloat16(bool isRelax);

  bool IsValid() const;

  //bool IsRelaxed() const { return mRelaxed; }

  const std::vector<OperandType>& operands() const {
    return operands_;
  }
  const std::vector<Operation>& operations() const {
    return operations_;
  }
  const std::vector<uint32_t>& input_indexes() const {
    return input_indexes_;
  }
  const std::vector<uint32_t>& output_indexes() const {
    return output_indexes_;
  }
  const std::vector<uint8_t>& operand_values() const {
    return operand_values_;
  }
  const std::vector<LargeValue>& large_values() const {
    return large_values_;
  }
  const std::vector<uint8_t>& small_values() const {
    return small_values_;
  }

  size_t NumPools() const {
    return pools_.size();
  }

 private:
  std::vector<OperandType> operands_;
  std::vector<Operation> operations_;
  std::vector<uint32_t> input_indexes_;
  std::vector<uint32_t> output_indexes_;
  std::vector<uint8_t> operand_values_;
  std::vector<LargeValue> large_values_;
  std::vector<uint8_t> small_values_;

  std::vector<Memory> pools_;
};

#endif  // define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_TYPES_H
