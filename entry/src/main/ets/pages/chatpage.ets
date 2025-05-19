import { Message } from '../model/Message';
import { StorageUtil } from '../utils/StorageUtil';
import { mindsporeUtil } from '../utils/MindsporeUtil';

import { mindSporeLite } from '@kit.MindSporeLiteKit'

interface MindsporeUtil {
  inference(text: string): Promise<string>;
}



// Add the ChatBubble component definition
@Component
struct ChatBubble {
  content: string = '';
  isUser: boolean = false;

  build() {
    Column() {
      Text(this.content)
        .fontSize(16)
        .backgroundColor(this.isUser ? '#DCF8C6' : '#FFFFFF')
        .borderRadius(10)
        .padding(10)
        .margin(5)
    }
    .alignItems(this.isUser ? HorizontalAlign.End : HorizontalAlign.Start)
    .width('100%')
  }
}

@Component
export struct ChatPage {
  @Prop convId: string = '';
  @State messages: Array<Message> = [];

  // Replace typeof with interface
  private storage: StorageUtil = new StorageUtil();
  private mindspore: MindsporeUtil = mindsporeUtil;

  aboutToAppear() {
    // Load messages when component appears
    this.loadMessages();
  }

  private loadMessages(): void {
    // Load messages from storage if available
    const savedMessages = this.storage.getMessages(this.convId);
    if (savedMessages && savedMessages.length > 0) {
      this.messages = savedMessages;
    }
  }

  build() {
    Column() {
      // Messages list
      List() {
        ForEach(this.messages, (msg: Message) => {
          ListItem() {
            ChatBubble({
              content: msg.content,
              isUser: msg.isUser
            })
          }
        }, (msg: Message) => msg.id)
      }
      .width('100%')
      .height('80%')

      // Input area
      Row() {
        TextInput({ placeholder: '请输入消息...' })
          .width('80%')
          .height(40)
          .onChange((value: string) => {
            this.inputText = value;
          })

        Button('发送')
          .width('20%')
          .height(40)
          .onClick(() => {
            if (this.inputText && this.inputText.trim().length > 0) {
              this.handleSend(this.inputText.trim());
              this.inputText = '';
            }
          })
      }
      .width('100%')
      .height('20%')
      .padding(10)
    }
    .width('100%')
    .height('100%')
  }

  @State private inputText: string = '';

  private async handleSend(text: string): Promise<void> {
    const userMsg = new Message(text, true);
    this.messages.push(userMsg);

    // MindSpore inference
    try {
      const aiResponse = await this.mindspore.inference(text);
      const aiMsg = new Message(aiResponse || '抱歉，我无法处理这个请求', false);
      this.messages.push(aiMsg);
    } catch (error) {
      // Handle error case
      const errorMsg = new Message('抱歉，发生了错误', false);
      this.messages.push(errorMsg);
    }

    // Save messages
    this.storage.saveMessages(this.convId, this.messages);
  }
}