declare module 'libentry.so' {
  export const mnn_wrapper: {
    CreateInterpreter: (modelPath: string) => number;
    CreateSession: (interpreterPtr: number, numThreads: number) => number;
    RunSession: (interpreterPtr: number, sessionPtr: number) => void;
    SetInputTensor: (interpreterPtr: number, sessionPtr: number, name: string, 
                    shape: number[], data: number[]) => void;
    GetOutputTensor: (interpreterPtr: number, sessionPtr: number, name: string) => number[];
    GetInputNames: (interpreterPtr: number, sessionPtr: number) => string[];
    GetOutputNames: (interpreterPtr: number, sessionPtr: number) => string[];
    ProcessImage: (interpreterPtr: number, sessionPtr: number, inputName: string, 
                 imageData: number[], width: number, height: number, channels: number) => number[];
  };
}