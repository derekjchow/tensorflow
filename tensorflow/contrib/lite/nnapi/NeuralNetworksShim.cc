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

#include <memory>
#include <vector>

struct Dummy {
  int whocares;
};

struct Memory {
  size_t size;
  int prot;
  int fd;
  size_t offset;
};

int ANeuralNetworksMemory_createFromFd(size_t size, int prot, int fd, size_t offset,
                                       ANeuralNetworksMemory** memory) {
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
  delete reinterpret_cast<Dummy*>(memory);
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
  *model = reinterpret_cast<ANeuralNetworksModel*>(new Dummy);
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
  delete reinterpret_cast<Dummy*>(model);
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   size_t offset, size_t length) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t* inputs, uint32_t outputCount,
                                      const uint32_t* outputs) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount,
                                                  const uint32_t* inputs, uint32_t outputCount,
                                                  const uint32_t* outputs) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model,
                                                          bool allow) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation) {
  *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(new Dummy);
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation) {
  delete reinterpret_cast<Dummy*>(compilation);
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution) {
  *execution = reinterpret_cast<ANeuralNetworksExecution*>(new Dummy);
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
  delete reinterpret_cast<Dummy*>(execution);
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type, const void* buffer,
                                      size_t length) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory, size_t offset,
                                                size_t length) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory, size_t offset,
                                                 size_t length) {
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event) {
  *event = reinterpret_cast<ANeuralNetworksEvent*>(new Dummy);
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
  delete reinterpret_cast<Dummy*>(event);
}
