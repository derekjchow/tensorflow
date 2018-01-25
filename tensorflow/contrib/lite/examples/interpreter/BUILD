package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "interpreter_example",
    srcs = ["interpreter_example.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/lite/python:interpreter",
        "//tensorflow/python:platform",
    ],
)
