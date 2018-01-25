/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CONTRIB_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
#define TENSORFLOW_CONTRIB_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_

#include <string>

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

namespace interpreter_wrapper {

class InterpreterWrapper {
 public:
  virtual ~InterpreterWrapper() = default;
  virtual bool AllocateTensors() = 0;
  virtual bool Invoke() = 0;

  virtual int NumInputs() const = 0;
  virtual std::string InputName(int i) const = 0;
  virtual int InputType(int i) const = 0;
  virtual void InputSize(int i, int** dims, int* size) const = 0;
  virtual void SetInputUint8(int i, uint8_t* data, int size) = 0;
  virtual void SetInputFloat32(int i, float* data, int size) = 0;

  virtual int NumOutputs() const = 0;
  virtual std::string OutputName(int i) const = 0;
  virtual int OutputType(int i) const = 0;
  virtual void OutputSize(int i, int** dims, int* size) const = 0;
  virtual void GetOutputUint8(int i, uint8_t** data, int* size) = 0;
  virtual void GetOutputFloat32(int i, float** data, int* size) = 0;
};

InterpreterWrapper* CreateWrapperCPP(const char* model_path);

}  // namespace interpreter_wrapper

#endif  // TENSORFLOW_CONTRIB_LITE_PYTHON_INTERPRETER_WRAPPER_INTERPRETER_WRAPPER_H_
