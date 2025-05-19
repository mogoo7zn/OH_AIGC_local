declare module 'libmnn_wrapper.so' {
  /**
   * Creates an MNN interpreter from a model file
   * @param modelPath Path to the model file
   * @returns Pointer to the interpreter as int64
   */
  export function CreateInterpreter(modelPath: string): number;

  /**
   * Creates an MNN session from an interpreter
   * @param interpreterPtr Pointer to the interpreter
   * @param numThreads Number of threads to use (default: 4)
   * @returns Pointer to the session as int64
   */
  export function CreateSession(interpreterPtr: number, numThreads?: number): number;

  /**
   * Runs an MNN session
   * @param interpreterPtr Pointer to the interpreter
   * @param sessionPtr Pointer to the session
   */
  export function RunSession(interpreterPtr: number, sessionPtr: number): void;
}

export {};