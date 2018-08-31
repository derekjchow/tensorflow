class CpuExecutor {
public:
    // Executes the model. The results will be stored at the locations
    // specified in the constructor.
    // The model must outlive the executor.  We prevent it from being modified
    // while this is executing.
    int Run(const Model& model, const Request& request,
            const std::vector<RunTimePoolInfo>& modelPoolInfos,
            const std::vector<RunTimePoolInfo>& requestPoolInfos) {
      for (const auto& operation : model.operations()) {
        int result = ExecuteOperation()
        if (result != ANEURALNETWORKS_NO_ERROR) {
          return result;
        }
      }
    }

private:
    // Runs one operation of the graph.
    int ExecuteOperation(const Operation& op) {
      const std::vector<uint32_t>& op_inputs = op.inputs;
      const std::vector<uint32_t>& op_outputs = op.outputs;
      bool success = false;

      switch (op.type) {
        case OperationType::CONV_2D:
          const size_t in_count = op_inputs.size();
          if ((in_count != 10 && in_count != 7)) {
              return ANEURALNETWORKS_BAD_DATA;
          }

          // int32_t padding_left, padding_right;
          // int32_t padding_top, padding_bottom;
          // int32_t stride_width, stride_height;
          // int32_t activation;
          // TODO(derekjchow): Finish me
          break;
      }
      return ANEURALNETWORKS_NO_ERROR;
    }
    // Decrement the usage count for the operands listed.  Frees the memory
    // allocated for any temporary variable with a count of zero.
    void freeNoLongerUsedOperands(const std::vector<uint32_t>& inputs);

    // The model and the request that we'll execute. Only valid while run()
    // is being executed.
    const Model* mModel = nullptr;
    const Request* mRequest = nullptr;

    // We're copying the list of all the dimensions from the model, as
    // these may be modified when we run the operatins.  Since we're
    // making a full copy, the indexes used in the operand description
    // stay valid.
    //    std::vector<uint32_t> mDimensions;
    // Runtime information about all the operands.
    std::vector<RunTimeOperandInfo> mOperands;
};
