import mnnService from './MnnService';
import fs from '@ohos.file.fs';
import { BusinessError } from '@kit.BasicServicesKit';

export class TextGenerationService {
  public isModelLoaded: boolean = false;
  private vocabulary: Map<string, number> = new Map();
  private idToToken: Map<number, string> = new Map();
  
  /**
   * Load model and vocabulary for text generation
   */
    async loadModel(modelPath: string, vocabPath: string): Promise<boolean> {
    try {
      this.isModelLoaded = mnnService.loadModel(modelPath);
      if (!this.isModelLoaded) {
        console.error(`Failed to load the MNN model from ${modelPath}`);
        return false;
      }

      // Load vocabulary using the fd path
      try {
        console.info(`Loading vocabulary from path: ${vocabPath}`);
        // fs.open can take a URI like "fd://<number>"
        const file = await fs.open(vocabPath, fs.OpenMode.READ_ONLY);
        // fs.readText can take a file descriptor (number)
        const textContent = await fs.readText(file.fd.toString());
        await fs.close(file);

        const tokens = textContent.split('\n');
        this.vocabulary.clear();
        this.idToToken.clear();
        tokens.forEach((token: string, index: number) => {
          const trimmedToken = token.trim();
          if (trimmedToken) {
            this.vocabulary.set(trimmedToken, index);
            this.idToToken.set(index, trimmedToken);
          }
        });
        console.info(`Vocabulary loaded successfully. Size: ${this.vocabulary.size}`);
      } catch (err) {
        const error = err as BusinessError;
        console.error(`Failed to load vocabulary from ${vocabPath}: ${error.message || JSON.stringify(err)}`);
        return false;
      }

      return true;
    } catch (error) {
      const err = error as BusinessError;
      console.error(`Error in TextGenerationService.loadModel: ${err.message || JSON.stringify(error)}`);
      return false;
    }
  }
  
  /**
   * Generate text from a prompt
   */
  generateText(prompt: string, maxLength: number = 100, callback?: (token: string) => void): string {
    if (!this.isModelLoaded) {
      console.error('Model not loaded');
      return '';
    }
    
    try {
      // Tokenize prompt (convert text to token IDs)
      const tokenIds = this.tokenize(prompt);
      
      // Set input shape and data
      const inputNames = mnnService.getInputNames();
      if (inputNames.length === 0) {
        console.error('No input tensors found');
        return '';
      }
      
      // Set input data - fix the mapping to ensure all are numbers
      const numericTokenIds = tokenIds.map((id: number) => Number(id));
      mnnService.setInput(inputNames[0], [1, tokenIds.length], numericTokenIds);
      
      // Run inference
      mnnService.runInference();
      
      // Get output
      const outputNames = mnnService.getOutputNames();
      if (outputNames.length === 0) {
        console.error('No output tensors found');
        return '';
      }
      
      // Get output tokens
      const outputLogits = mnnService.getOutput(outputNames[0]);
      
      // Simple text generation (in practice, you would implement proper token decoding)
      let generatedText = prompt;
      
      // Get the token with highest probability for each position
      for (let i = 0; i < maxLength; i++) {
        // Get predicted token ID
        const nextTokenId = this.getNextToken(outputLogits);
        
        // Convert token ID back to text
        const token = this.idToToken.get(nextTokenId) || '';
        if (token === '<EOS>' || token === '') break;
        
        generatedText += token;
        
        // Call callback for streaming if provided
        if (callback) {
          callback(token);
        }
      }
      
      return generatedText;
    } catch (error) {
      console.error(`Error in generateText: ${error instanceof Error ? error.message : String(error)}`);
      return '';
    }
  }
  
  /**
   * Generate text with streaming output
   */
  async generateTextStreaming(prompt: string, maxLength: number = 100, 
                             callback: (token: string) => void): Promise<string> {
    let fullText = prompt;
    
    return new Promise<string>((resolve: (value: string) => void) => {
      this.generateText(prompt, maxLength, (token: string) => {
        fullText += token;
        callback(token);
      });
      
      resolve(fullText);
    });
  }
  
  private tokenize(text: string): Array<number> {
    // String input must be converted to words, then to token IDs
    return text.split(' ').map((word: string) => {
        const tokenId = this.vocabulary.get(word);
        return tokenId !== undefined ? tokenId : 0;
    });
  }
  
  private getNextToken(logits: Array<number>): number {
    // Simplified implementation - find token with highest probability
    let maxIndex = 0;
    let maxValue = -Infinity;
    
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxValue) {
        maxValue = logits[i];
        maxIndex = i;
      }
    }
    
    return maxIndex;
  }
}

export default new TextGenerationService();