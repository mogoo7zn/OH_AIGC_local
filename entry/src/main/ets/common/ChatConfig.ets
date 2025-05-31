import { MNNModelManager } from './MNNModelManager';
import { ChatMessage, ChatRole } from '@changwei/chatui';

/**
 * Chat configuration with MNN model integration
 */
export class ChatConfig {
  private modelManager: MNNModelManager = new MNNModelManager();
  private modelLoaded: boolean = false;
  private modelPath: string = 'internal://app/models/my_model.mnn'; // Update with your model path

  constructor() {
    this.initModel();
  }

  private async initModel(): Promise<void> {
    try {
      this.modelLoaded = await this.modelManager.loadModel(this.modelPath);
      console.info('Model initialization ' + (this.modelLoaded ? 'successful' : 'failed'));
    } catch (error) {
      console.error('Model initialization error:', error);
    }
  }

  /**
   * Process message with MNN model
   * @param text Input message text
   * @returns Promise with response from model
   */
  async processMessage(text: string): Promise<string> {
    if (!this.modelLoaded) {
      return 'Model not loaded. Please try again later.';
    }
    
    try {
      return await this.modelManager.inferWithText(text);
    } catch (error) {
      console.error('Error processing message with model:', error);
      return 'Sorry, I encountered an error processing your message.';
    }
  }

  /**
   * Handle incoming messages and generate responses using MNN model
   * @param ctl Chat controller
   * @param message User message
   */
  async handleMessage(ctl: any, message: ChatMessage): Promise<void> {
    // Post user message to chat
    ctl.postMessage(message);
    
    // Show typing indicator
    ctl.setTyping(true);
    
    try {
      // Process message with MNN model
      const response = await this.processMessage(message.content);
      
      // Create response message
      const responseMessage = new ChatMessage({
        role: ChatRole.Assistant,
        content: response
      });
      
      // Post response to chat
      ctl.postMessage(responseMessage);
    } catch (error) {
      console.error('Error handling message:', error);
      
      // Send error message
      const errorMessage = new ChatMessage({
        role: ChatRole.Assistant,
        content: 'Sorry, I encountered an error processing your request.'
      });
      
      ctl.postMessage(errorMessage);
    } finally {
      // Hide typing indicator
      ctl.setTyping(false);
    }
  }
}