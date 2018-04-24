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

// Contains all the entry points to the C Neural Networks API.
// We do basic validation of the operands and then call the class
// that implements the functionality.

#define LOG_TAG "NeuralNetworks"

#include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>


#define LOG(x) std::cerr

namespace {
uint32_t AlignBytesNeeded(uint32_t index, size_t length) {
    uint32_t pattern;
    if (length < 2) {
        pattern = 0; // No alignment necessary
    } else if (length < 4) {
        pattern = 1; // Align on 2-byte boundary
    } else {
        pattern = 3; // Align on 4-byte boundary
    }
    uint32_t extra = (~(index - 1)) & pattern;
    return extra;
}

}

struct Dummy {
  int whocares;
};

struct Memory {
  size_t size;
  int prot;
  int fd;
  size_t offset;
};

enum class OperandLifeTime {
  TEMPORARY_VARIABLE,
  MODEL_INPUT,
  MODEL_OUTPUT,
  CONSTANT_COPY,
  CONSTANT_REFERENCE,
  NO_VALUE,
};

struct DataLocation {
  uint32_t pool_index;
  uint32_t offset;
  uint32_t length;
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

// This is a HIDL struct
class Model {
 public:
  Model() {
    LOG(ERROR) << __FUNCTION__ << "\n";
  }
  ~Model() {
    LOG(ERROR) << __FUNCTION__ << "\n";
  }

  int Finish() {
    // TODO(derekjchow): Check for completed model
    // TODO(derekjchow): Check for invalid model

    // NB(derekjchow): Large values are copied in Android NN API here to be sent
    // over binder, but let's handle this in the compilation.

    LOG(ERROR) << __PRETTY_FUNCTION__ << "\n";
    return ANEURALNETWORKS_NO_ERROR;
  }

  int AddOperand(const OperandType& type) {
    LOG(ERROR) << __FUNCTION__ << "\n";
    // TODO(derekjchow): We could make this a zero copy thing, but I don't think
    // it's conformant to the NN API spec. Either way, this is just done at
    // compile time so it shouldn't be a big deal... probably.
    operands_.push_back(type);
    return ANEURALNETWORKS_NO_ERROR;
  }

  int SetOperandValue(uint32_t index, const void* buffer, size_t length) {
    LOG(ERROR) << __FUNCTION__ << "\n";
    OperandType& operand = operands_[index];
    if (!buffer) {
      if (length) {
        return ANEURALNETWORKS_BAD_DATA;
      }

      operand.lifetime = OperandLifeTime::NO_VALUE;
      operand.location = {.pool_index = 0, .offset = 0, .length = 0};
      return ANEURALNETWORKS_NO_ERROR;
    }

    // TODO(derekjchow): More length checks.
    if (length > 0xFFFFFFFF) {
      LOG(ERROR) << "ANeuralNetworksModel_setOperandValue value length of "
                 << length << " exceeds max size";
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (length < ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
      // TODO(derekjchow): I think NN API allocates one private pool for it's
      // own use. If this is the case, figure out where this is added.

      uint32_t existing_size = static_cast<uint32_t>(small_values_.size());
      uint32_t extra_bytes = AlignBytesNeeded(existing_size, length);
      small_values_.resize(existing_size + extra_bytes + length);
      operand.lifetime = OperandLifeTime::CONSTANT_COPY;
      operand.location = {
          .pool_index = 0,
          .offset = existing_size + extra_bytes,
          .length = length};
      memcpy(&small_values_[operand.location.offset], buffer, length);

      return ANEURALNETWORKS_NO_ERROR;
    }

    operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
    typedef decltype(operand.location.pool_index) PoolIndexType;
    typedef decltype(operand.location.offset) OffsetType;
    operand.location = {
        .pool_index = ~PoolIndexType(0),
        .offset = ~OffsetType(0),
        .length = length
    };

    large_values_.push_back({.operand_index = index, .buffer = buffer});

    return ANEURALNETWORKS_NO_ERROR;
  }

  int SetOperandValueFromMemory(uint32_t index, const Memory* memory,
                                 uint32_t offset, size_t length) {
    LOG(ERROR) << __FUNCTION__ << "\n";

    pools_.push_back(*memory);
    uint32_t pool_index = pools_.size() - 1;

    OperandType& operand = operands_[index];
    operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
    operand.location = {
        .pool_index = pool_index, .offset = offset, .length = length};

    return ANEURALNETWORKS_NO_ERROR;
  }

  void AddOperation(ANeuralNetworksOperationType type,
                    const std::vector<uint32_t>& inputs,
                    const std::vector<uint32_t>& outputs) {
    LOG(ERROR) << __FUNCTION__ << "\n";
    operations_.push_back({.type = type, .inputs = inputs, .outputs = outputs});
  }

  void IdentifyInputsAndOutputs(const std::vector<uint32_t>& inputs,
                                const std::vector<uint32_t>& outputs) {
    LOG(ERROR) << __FUNCTION__ << "\n";
    // TODO(derekjchow): Make this function pass by value and std::swap?
    input_indexes_ = inputs;
    output_indexes_ = outputs;
  }

  //void RelaxComputationFloat32toFloat16(bool isRelax);

  bool IsValid() const {
    // TODO(derekjchow): Check me
    return true;
  }
  //bool IsRelaxed() const { return mRelaxed; }
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

int ANeuralNetworksMemory_createFromFd(size_t size, int prot, int fd, size_t offset,
                                       ANeuralNetworksMemory** memory) {
    LOG(ERROR) << __FUNCTION__ << "\n";
    Memory* m = new Memory;
    m->size = size;
    m->prot = prot;
    m->fd = fd;
    m->offset = offset;

    // TODO(derekjchow): Do some memory mapping here yo

    *memory = reinterpret_cast<ANeuralNetworksMemory*>(m);
    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  delete reinterpret_cast<Dummy*>(memory);
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
  *model = reinterpret_cast<ANeuralNetworksModel*>(new Model);
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
  delete reinterpret_cast<Model*>(model);
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
  Model* m = reinterpret_cast<Model*>(model);
  return m->Finish();
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type) {
  OperandType operand;
  operand.type = type->type;
  operand.scale = type->scale;
  operand.zero_point = type->zeroPoint;

  if (type->dimensionCount) {
    operand.dimensions = std::vector<uint32_t>(
        type->dimensions, type->dimensions + type->dimensionCount);
  }

  Model* m = reinterpret_cast<Model*>(model);
  return m->AddOperand(operand);
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length) {
  Model* m = reinterpret_cast<Model*>(model);
  return m->SetOperandValue(index, buffer, length);
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   size_t offset, size_t length) {
  Model* m = reinterpret_cast<Model*>(model);
  const Memory* mem = reinterpret_cast<const Memory*>(memory);
  return m->SetOperandValueFromMemory(index, mem, offset, length);
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t* inputs, uint32_t outputCount,
                                      const uint32_t* outputs) {
  std::vector<uint32_t> inputs_vec(inputs, inputs + inputCount);
  std::vector<uint32_t> outputs_vec(outputs, outputs + outputCount);
  Model* m = reinterpret_cast<Model*>(model);
  m->AddOperation(type, inputs_vec, outputs_vec);
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount,
                                                  const uint32_t* inputs, uint32_t outputCount,
                                                  const uint32_t* outputs) {
  // TODO(derekjchow): Check for null ptr

  // TODO(derekjchow): Zero copy this
  std::vector<uint32_t> inputs_vec(inputs, inputs + inputCount);
  std::vector<uint32_t> outputs_vec(outputs, outputs + outputCount);
  Model* m = reinterpret_cast<Model*>(model);
  m->IdentifyInputsAndOutputs(inputs_vec, outputs_vec);
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model,
                                                          bool allow) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(new Dummy);
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  delete reinterpret_cast<Dummy*>(compilation);
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  *execution = reinterpret_cast<ANeuralNetworksExecution*>(new Dummy);
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  delete reinterpret_cast<Dummy*>(execution);
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type, const void* buffer,
                                      size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory, size_t offset,
                                                size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory, size_t offset,
                                                 size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  *event = reinterpret_cast<ANeuralNetworksEvent*>(new Dummy);
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  delete reinterpret_cast<Dummy*>(event);
}
