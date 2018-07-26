licenses(["notice"])  # Apache 2.0

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "nnapi_lib",
    hdrs = [
        "NeuralNetworksShim.h",
        "internal/NeuralNetworksInterfaces.h",
        "internal/NeuralNetworksTypes.h",
    ],
    srcs = [
        "NeuralNetworksShim.cc",
        "internal/NeuralNetworksInferfacesInternal.cc",
    ],
    #linkopts = ["-ldl"],
)
