import { mnn_wrapper } from 'libentry.so';

export class MnnService {
  private interpreter: number = 0;
  private session: number = 0;
  
  /**
   * Load an MNN model from file
   */
  loadModel(modelPath: string, numThreads: number = 4): boolean {
    try {
      this.interpreter = mnn_wrapper.CreateInterpreter(modelPath);
      if (this.interpreter === 0) {
        console.error(`Failed to load model at ${modelPath}`);
        return false;
      }
      
      this.session = mnn_wrapper.CreateSession(this.interpreter, numThreads);
      if (this.session === 0) {
        console.error('Failed to create session');
        return false;
      }
      
      return true;
    } catch (error) {
      console.error(`Error loading model: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }
  
  /**
   * Get names of input tensors
   */
  getInputNames(): Array<string> {
    if (!this.interpreter || !this.session) return [];
    return mnn_wrapper.GetInputNames(this.interpreter, this.session);
  }
  
  /**
   * Get names of output tensors
   */
  getOutputNames(): Array<string> {
    if (!this.interpreter || !this.session) return [];
    return mnn_wrapper.GetOutputNames(this.interpreter, this.session);
  }
  
  /**
   * Set input tensor data
   */
  setInput(name: string, shape: Array<number>, data: Array<number>): void {
    if (!this.interpreter || !this.session) return;
    mnn_wrapper.SetInputTensor(this.interpreter, this.session, name, shape, data);
  }
  
  /**
   * Run inference on the model
   */
  runInference(): void {
    if (!this.interpreter || !this.session) return;
    mnn_wrapper.RunSession(this.interpreter, this.session);
  }
  
  /**
   * Get output tensor data
   */
  getOutput(name: string): Array<number> {
    if (!this.interpreter || !this.session) return [];
    return mnn_wrapper.GetOutputTensor(this.interpreter, this.session, name);
  }
}

export default new MnnService();