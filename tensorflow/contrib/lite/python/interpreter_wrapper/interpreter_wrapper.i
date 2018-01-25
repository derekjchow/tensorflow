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

%include "std_string.i"


%{
#define SWIG_FILE_WITH_INIT
#include "tensorflow/contrib/lite/python/interpreter_wrapper/interpreter_wrapper.h"
%}

%include "tensorflow/contrib/lite/python/interpreter_wrapper/numpy.i"
%init %{
import_array();
%}


%apply (unsigned char* IN_ARRAY1, int DIM1) {(uint8_t* data, int size)};
%apply (unsigned char** ARGOUTVIEWM_ARRAY1, int* DIM1) {(uint8_t** data, int* size)};

%apply (float* IN_ARRAY1, int DIM1) {(float* data, int size)};
%apply (float** ARGOUTVIEWM_ARRAY1, int* DIM1) {(float** data, int* size)};

%apply (int** ARGOUTVIEWM_ARRAY1, int* DIM1) {(int** dims, int* size)};


%include "tensorflow/contrib/lite/python/interpreter_wrapper/interpreter_wrapper.h"
