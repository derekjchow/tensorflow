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
#include "tensorflow/contrib/lite/python/interpreter_wrapper/interpreter_wrapper.h"

#include <string>

#include "tensorflow/core/platform/logging.h"

#if PY_MAJOR_VERSION >= 3
#define PY_TO_CPPSTRING PyBytes_AsStringAndSize
#define CPP_TO_PYSTRING PyBytes_FromStringAndSize
#else
#define PY_TO_CPPSTRING PyString_AsStringAndSize
#define CPP_TO_PYSTRING PyString_FromStringAndSize
#endif

namespace interpreter_wrapper {

namespace {
std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel* model,
    const tflite::ops::builtin::BuiltinOpResolver& resolver) {
  if (!model) {
    return nullptr;
  }

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (interpreter) {
    for (const auto input_index : interpreter->inputs()) {
      const auto* tensor = interpreter->tensor(input_index);
      CHECK(tensor);
      const auto* dims = tensor->dims;
      if (!dims) {
        continue;
      }

      std::vector<int> input_dims(dims->size);
      for (int i = 0; i < dims->size; i++) {
        input_dims[i] = dims->data[i];
      }
      interpreter->ResizeInputTensor(input_index, input_dims);
    }
  }
  return interpreter;
}
}  // namespace

class InterpreterWrapperImpl : public InterpreterWrapper {
 public:
  explicit InterpreterWrapperImpl(
      std::unique_ptr<tflite::FlatBufferModel> model);
  ~InterpreterWrapperImpl() override{};

  bool AllocateTensors() override;
  bool Invoke() override;
  std::string InputName(int i) const override;
  std::string OutputName(int i) const override;
  int NumInputs() const override;
  int NumOutputs() const override;

  void InputSize(int i, int** dims, int* size) const override;
  void OutputSize(int i, int** dims, int* size) const override;

  int InputType(int i) const override;
  int OutputType(int i) const override;

  void SetInputUint8(int i, uint8_t* data, int size) override;
  void GetOutputUint8(int i, uint8_t** data, int* size) override;

  void SetInputFloat32(int i, float* data, int size) override;
  void GetOutputFloat32(int i, float** data, int* size) override;

 private:
  template<typename T, TfLiteType expected_type>
  void SetInput(int i, const T* data, int size);

  template<typename T, TfLiteType expected_type>
  void GetOutput(int i, T** data, int* size);

  const std::unique_ptr<tflite::FlatBufferModel> model_;
  tflite::ops::builtin::BuiltinOpResolver resolver_;
  const std::unique_ptr<tflite::Interpreter> interpreter_;
};

InterpreterWrapperImpl::InterpreterWrapperImpl(
    std::unique_ptr<tflite::FlatBufferModel> model)
    : model_(std::move(model)),
      resolver_(tflite::ops::builtin::BuiltinOpResolver()),
      interpreter_(CreateInterpreter(model_.get(), resolver_)) {}

bool InterpreterWrapperImpl::AllocateTensors() {
  if (!interpreter_) {
    LOG(ERROR) << "Cannot allocate tensors: invalid interpreter.";
    return false;
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Unable to allocate tensors.";
  }

  return (interpreter_->AllocateTensors() == kTfLiteOk);
}

bool InterpreterWrapperImpl::Invoke() {
  return interpreter_ ? (interpreter_->Invoke() == kTfLiteOk) : false;
}

std::string InterpreterWrapperImpl::InputName(int i) const {
  if (!interpreter_ || i >= interpreter_->inputs().size()) {
    return "";
  }

  return interpreter_->GetInputName(i);
}

std::string InterpreterWrapperImpl::OutputName(int i) const {
  if (!interpreter_ || i >= interpreter_->outputs().size()) {
    return "";
  }

  return interpreter_->GetOutputName(i);
}

int InterpreterWrapperImpl::NumInputs() const {
  return interpreter_ ? interpreter_->inputs().size() : -1;
}

int InterpreterWrapperImpl::NumOutputs() const {
  return interpreter_ ? interpreter_->outputs().size() : -1;
}

void InterpreterWrapperImpl::InputSize(int i, int** dims, int* size) const {
  if (!interpreter_ || i >= interpreter_->inputs().size()) {
    *size = 0;
    return;
  }

  int input_index = interpreter_->inputs()[i];
  const auto* input_tensor = interpreter_->tensor(input_index);
  const int tensor_size = input_tensor->dims->size;

  // Lifetime of *dims will be managed by SWIG binding.
  *dims = static_cast<int*>(malloc(tensor_size * sizeof(int)));
  *size = tensor_size;
  memcpy(*dims, input_tensor->dims->data, tensor_size * sizeof(int));
}

void InterpreterWrapperImpl::OutputSize(int i, int** dims, int* size) const {
  if (!interpreter_ || i >= interpreter_->outputs().size()) {
    *size = 0;
    return;
  }

  int output_index = interpreter_->outputs()[i];
  const auto* output_tensor = interpreter_->tensor(output_index);
  const int tensor_size = output_tensor->dims->size;

  // Lifetime of *dims will be managed by SWIG binding.
  *dims = static_cast<int*>(malloc(tensor_size * sizeof(int)));
  *size = tensor_size;
  memcpy(*dims, output_tensor->dims->data, tensor_size * sizeof(int));
}

int InterpreterWrapperImpl::InputType(int i) const {
  if (!interpreter_ || i >= interpreter_->inputs().size()) {
    return kTfLiteNoType;
  }

  int input_index = interpreter_->inputs()[i];
  const auto* input_tensor = interpreter_->tensor(input_index);
  return input_tensor->type;
}

int InterpreterWrapperImpl::OutputType(int i) const {
  if (!interpreter_ || i >= interpreter_->outputs().size()) {
    return kTfLiteNoType;
  }

  int output_index = interpreter_->outputs()[i];
  const auto* output_tensor = interpreter_->tensor(output_index);
  return output_tensor->type;
}

template<typename T, TfLiteType expected_type>
void InterpreterWrapperImpl::SetInput(int i, const T* data, int size) {
  if (!interpreter_) {
    LOG(ERROR) << "Invalid interpreter.";
    return;
  }

  if (i >= interpreter_->inputs().size()) {
    LOG(ERROR) << "Invalid tensor index: " << i
               << " exceeds max input tensor index "
               << interpreter_->inputs().size();
    return;
  }

  int input_index = interpreter_->inputs()[i];
  const auto* input_tensor = interpreter_->tensor(input_index);
  if (size*sizeof(T) != input_tensor->bytes) {
    LOG(ERROR) << "Cannot set input: Tensor size mismatch";
    return;
  }

  if (input_tensor->type != expected_type) {
    LOG(ERROR) << "Cannot set input: Bad tensor type";
    return;
  }

  T* input_buffer = interpreter_->typed_input_tensor<T>(i);
  if (!input_buffer) {
    LOG(ERROR) << "Cannot set input: Empty input tensor: " << i;
    return;
  }

  memcpy(input_buffer, data, size*sizeof(T));
}

void InterpreterWrapperImpl::SetInputUint8(int i, uint8_t* data, int size) {
  SetInput<uint8_t, kTfLiteUInt8>(i, data, size);
}

void InterpreterWrapperImpl::SetInputFloat32(int i, float* data, int size) {
  SetInput<float, kTfLiteFloat32>(i, data, size);
}

template<typename T, TfLiteType expected_type>
void InterpreterWrapperImpl::GetOutput(int i, T** data, int* size) {
  if (!interpreter_) {
    LOG(ERROR) << "Invalid interpreter.";
    return;
  }

  if (i >= interpreter_->outputs().size()) {
    LOG(ERROR) << "Invalid tensor index: " << i
               << " exceeds max input tensor index "
               << interpreter_->inputs().size();
    return;
  }

  int output_index = interpreter_->outputs()[i];
  const auto* output_tensor = interpreter_->tensor(output_index);
  const int tensor_size = output_tensor->bytes;
  if (tensor_size <= 0) {
    LOG(ERROR) << "Invalid output tensor size";
    *size = 0;
    return;
  }

  if (output_tensor->type != expected_type) {
    LOG(ERROR) << "Bad tensor type";
    *size = 0;
    return;
  }

  T* output_buffer = interpreter_->typed_output_tensor<T>(i);
  if (!output_buffer) {
    LOG(ERROR) << "Empty output buffer: " << i;
    *size = 0;
    return;
  }

  // Lifetime of *data will be managed by SWIG binding.
  *data = static_cast<T*>(malloc(tensor_size));
  *size = tensor_size / sizeof(T);

  memcpy(*data, output_buffer, tensor_size);
}

void InterpreterWrapperImpl::GetOutputUint8(int i, uint8_t** data, int* size) {
  GetOutput<uint8_t, kTfLiteUInt8>(i, data, size);
}

void InterpreterWrapperImpl::GetOutputFloat32(int i, float** data, int* size) {
  GetOutput<float, kTfLiteFloat32>(i, data, size);
}

InterpreterWrapper* CreateWrapperCPP(const char* model_path) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path);
  return model ? new InterpreterWrapperImpl(std::move(model)) : nullptr;
}

}  // namespace interpreter_wrapper
