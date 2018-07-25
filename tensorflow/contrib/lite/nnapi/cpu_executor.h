/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_COMMON_CPU_EXECUTOR_H
#define ANDROID_ML_NN_COMMON_CPU_EXECUTOR_H

// TODO(derekjchow): Move some internals into common header file?

#include <algorithm>
#include <vector>

namespace android {
namespace nn {

// This is a HIDL struct
struct RequestArgument {
    bool has_no_value;
    DataLocation location;
    std::vector<uint32_t> dimensions;
};

// This is a HIDL struct
struct Request {
    std::vector<RequestArgument> inputs;
    std::vector<RequestArgument> outputs;
    std::vector<Memory> pools;
};

// This class is used to execute a model on the CPU.
class CpuExecutor {
 public:
  int Run(const Model& model, const Request& request,
          const std::vector<Memory>& model_pools,
          const std::vector<Memory>& request_pools) {
    // Because this totally happens haha.
    return ANEURALNETWORKS_NO_ERROR;
  }

 private:
  int ExecuteOperation(const Operation& operation) {
    return ANEURALNETWORKS_NO_ERROR;
  }
};

namespace {

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
  // TODO: Check buffer is at least as long as size of data.
  T* data = reinterpret_cast<T*>(info.buffer);
  return data[0];
}

inline bool IsNullInput(const RunTimeOperandInfo *input) {
    return input->lifetime == OperandLifeTime::NO_VALUE;
}

inline int NumInputsWithValues(const Operation &operation,
                               std::vector<RunTimeOperandInfo> &operands) {
  const std::vector<uint32_t> &inputs = operation.inputs;
  return std::count_if(inputs.begin(), inputs.end(),
                       [&operands](uint32_t i) {
                         return !IsNullInput(&operands[i]);
                       });
}

inline int NumOutputs(const Operation &operation) {
  return operation.outputs.size();
}

inline size_t NumDimensions(const RunTimeOperandInfo *operand) {
  return operand->shape().dimensions.size();
}

inline uint32_t SizeOfDimension(const RunTimeOperandInfo *operand, int i) {
  return operand->shape().dimensions[i];
}

inline RunTimeOperandInfo *GetInput(const Operation &operation,
                                    std::vector<RunTimeOperandInfo> &operands,
                                    int index) {
  return &operands[operation.inputs[index]];
}

inline RunTimeOperandInfo *GetOutput(const Operation &operation,
                                     std::vector<RunTimeOperandInfo> &operands,
                                     int index) {
  return &operands[operation.outputs[index]];
}

}  // anonymous namespace

} // namespace nn
} // namespace android

#endif // ANDROID_ML_NN_COMMON_CPU_EXECUTOR_H
