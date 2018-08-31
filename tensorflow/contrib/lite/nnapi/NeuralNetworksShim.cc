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
#include "tensorflow/contrib/lite/nnapi/internal/NeuralNetworksInterfaces.h"
#include "tensorflow/contrib/lite/nnapi/internal/NeuralNetworksTypes.h"

#include <atomic>
#include <cstring>
#include <functional>
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

}  // namespace

struct Dummy {
  int whocares;
};

Model::Model() {
  LOG(ERROR) << __FUNCTION__ << "\n";
}

Model::~Model() {
  LOG(ERROR) << __FUNCTION__ << "\n";
}

int Model::Finish() {
  // TODO(derekjchow): Check for completed model
  // TODO(derekjchow): Check for invalid model

  // NB(derekjchow): Large values are copied in Android NN API here to be sent
  // over binder, but let's handle this in the compilation.

  LOG(ERROR) << __PRETTY_FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int Model::AddOperand(const OperandType& type) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  // TODO(derekjchow): We could make this a zero copy thing, but I don't think
  // it's conformant to the NN API spec. Either way, this is just done at
  // compile time so it shouldn't be a big deal... probably.
  operands_.push_back(type);
  return ANEURALNETWORKS_NO_ERROR;
}

int Model::SetOperandValue(uint32_t index, const void* buffer, size_t length) {
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
        .length = static_cast<uint32_t>(length)
    };
    memcpy(&small_values_[operand.location.offset], buffer, length);

    return ANEURALNETWORKS_NO_ERROR;
  }

  operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
  typedef decltype(operand.location.pool_index) PoolIndexType;
  typedef decltype(operand.location.offset) OffsetType;
  operand.location = {
      .pool_index = ~PoolIndexType(0),
      .offset = ~OffsetType(0),
      .length = static_cast<uint32_t>(length)
  };

  large_values_.push_back({.operand_index = index, .buffer = buffer});

  return ANEURALNETWORKS_NO_ERROR;
}

int Model::SetOperandValueFromMemory(uint32_t index, const Memory* memory,
                                     uint32_t offset, size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";

  pools_.push_back(*memory);
  uint32_t pool_index = pools_.size() - 1;

  OperandType& operand = operands_[index];
  operand.lifetime = OperandLifeTime::CONSTANT_REFERENCE;
  operand.location = {
      .pool_index = pool_index,
      .offset = offset,
      .length = static_cast<uint32_t>(length)
  };

  return ANEURALNETWORKS_NO_ERROR;
}

void Model::AddOperation(ANeuralNetworksOperationType type,
                         std::vector<uint32_t> inputs,
                         std::vector<uint32_t> outputs) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  operations_.push_back({
      .type = type,
      .inputs = std::move(inputs),
      .outputs = std::move(outputs)});
}

void Model::IdentifyInputsAndOutputs(std::vector<uint32_t> inputs,
                                     std::vector<uint32_t> outputs) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  input_indexes_ = std::move(inputs);
  output_indexes_ = std::move(outputs);
}

//void Model::RelaxComputationFloat32toFloat16(bool isRelax);

bool Model::IsValid() const {
  // TODO(derekjchow): Check me
  return true;
}

//bool Model::IsRelaxed() const { return mRelaxed; }


class Compilation {
 public:
  Compilation() = delete;

  Compilation(Model* model)
    : model_(model) {
    LOG(ERROR) << __FUNCTION__ << "\n";
  }

  ~Compilation() {
    LOG(ERROR) << __FUNCTION__ << "\n";
  }

  int SetPreference(int32_t preference) {
    // We're not fancy enough to determine if we want to run fast or slow. Just
    // nod and claim that we care.
    LOG(ERROR) << __FUNCTION__ << "\n";
    return ANEURALNETWORKS_NO_ERROR;
  }

  int Finish() {
    // TODO(derekjchow): We should do paritioning and call the HAL here:
    // IDevice::getCapabilities
    // IDevice::getSupportedOperations
    // IDevice::prepareModel
    // For now we're gonna be really dumb and get atomically request the model
    // get compiled. If he dies he dies.

    // TODO(derekjchow): "Get" driver static function
    proto_nnapi::NNDriver* driver = proto_nnapi::GetDriver();
    std::shared_ptr<proto_nnapi::NNPreparedModel> compiled_model;
    ErrorStatus status = ErrorStatus::NONE;
    // TODO(derekjchow): Use a better synchronization than atomic bool.
    // Or even better, make the user space API asynchronous.
    std::atomic<bool> done(false);
    auto callback = [&compiled_model, &status, &done](
        ErrorStatus arg_status,
        std::shared_ptr<proto_nnapi::NNPreparedModel> arg_compiled_model) {
      status = arg_status;
      compiled_model = std::move(arg_compiled_model);
      done.store(true);
    };
    driver->PrepareModel_1_1(
        *model_, ExecutionPreference::SUSTAINED_SPEED, callback);
    while (!done.load()) {}

    compiled_model_ = compiled_model;

    // TODO(derekjchow): Return error based on actual result.
    LOG(ERROR) << __PRETTY_FUNCTION__ << "\n";
    return ANEURALNETWORKS_NO_ERROR;
  }

  proto_nnapi::NNPreparedModel* compiled_model() {
    return compiled_model_.get();
  }

 private:
  std::shared_ptr<proto_nnapi::NNPreparedModel> compiled_model_;
  Model* const model_;
};

class Execution {
 public:
  class Event {
   public:
    explicit Event(const Execution* execution)
      : execution_(execution) {}

    int Wait() const {
      return ANEURALNETWORKS_NO_ERROR;
    }

    ~Event() {
      // TODO(derekjchow): What if computation hasn't finished yet?
    }

   private:
    const Execution* const execution_;
  };

  Execution(Compilation* compilation)
    : compilation_(compilation) {}
  ~Execution() {}

  int SetInput(int32_t index, const ANeuralNetworksOperandType* type,
               const void* buffer, size_t length) {
    return ANEURALNETWORKS_NO_ERROR;
  }

  int SetInputFromMemory(int32_t index, const ANeuralNetworksOperandType* type,
                         const Memory* memory, size_t offset, size_t length) {
    return ANEURALNETWORKS_NO_ERROR;
  }

  int SetOutput(int32_t index, const ANeuralNetworksOperandType* type,
                void* buffer, size_t length) {
    return ANEURALNETWORKS_NO_ERROR;
  }

  int SetOutputFromMemory(int32_t index,
                          const ANeuralNetworksOperandType* type,
                          const Memory* memory,
                          size_t offset,
                          size_t length) {
    return ANEURALNETWORKS_NO_ERROR;
  }

  Event* StartCompute() {
    proto_nnapi::NNPreparedModel* compiled_model =
        compilation_->compiled_model();
    Request request;

    auto callback = [](ErrorStatus e) {};

    compiled_model->Execute(request, callback);
    return new Event(this);
  }

 private:
  Compilation* const compilation_;
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
  m->AddOperation(type, std::move(inputs_vec), std::move(outputs_vec));
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount,
                                                  const uint32_t* inputs, uint32_t outputCount,
                                                  const uint32_t* outputs) {
  // TODO(derekjchow): Check for null ptr
  std::vector<uint32_t> inputs_vec(inputs, inputs + inputCount);
  std::vector<uint32_t> outputs_vec(outputs, outputs + outputCount);
  Model* m = reinterpret_cast<Model*>(model);
  m->IdentifyInputsAndOutputs(std::move(inputs_vec), (outputs_vec));
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model,
                                                          bool allow) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation) {
  Model* m = reinterpret_cast<Model*>(model);
  *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(
      new Compilation(m));
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation) {
  delete reinterpret_cast<Compilation*>(compilation);
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference) {
  return reinterpret_cast<Compilation*>(compilation)->SetPreference(preference);
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation) {
  return reinterpret_cast<Compilation*>(compilation)->Finish();
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  *execution = reinterpret_cast<ANeuralNetworksExecution*>(
      new Execution(reinterpret_cast<Compilation*>(compilation)));
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  delete reinterpret_cast<Execution*>(execution);
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type, const void* buffer,
                                      size_t length) {
  if (!execution || (!buffer && length != 0)) {
      LOG(ERROR) << "ANeuralNetworksExecution_setInput passed a nullptr";
      return ANEURALNETWORKS_UNEXPECTED_NULL;
  }
  LOG(ERROR) << __FUNCTION__ << "\n";
  auto e = reinterpret_cast<Execution*>(execution);
  return e->SetInput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory, size_t offset,
                                                size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  Execution* e = reinterpret_cast<Execution*>(execution);
  const Memory* m = reinterpret_cast<const Memory*>(memory);
  return e->SetInputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  auto e = reinterpret_cast<Execution*>(execution);
  return e->SetOutput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory, size_t offset,
                                                 size_t length) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  Execution* e = reinterpret_cast<Execution*>(execution);
  const Memory* m = reinterpret_cast<const Memory*>(memory);
  return e->SetOutputFromMemory(index, type, m, offset, length);
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  // TODO(derekjchow): Nullptr checks
  Execution* e = reinterpret_cast<Execution*>(execution);
  *event = reinterpret_cast<ANeuralNetworksEvent*>(e->StartCompute());
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  return reinterpret_cast<Execution::Event*>(event)->Wait();
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
  LOG(ERROR) << __FUNCTION__ << "\n";
  delete reinterpret_cast<Execution::Event*>(event);
}
