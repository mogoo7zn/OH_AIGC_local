import util from '@ohos.util';

export class Conversation {
  id: string = '';
  title: string = '';
  messages: Array<string> = [];
  lastMessageTime: Date = new Date();

  constructor() {
    this.id = util.generateRandomUUID();
    this.title = "新对话";
    this.messages = [];
    this.lastMessageTime = new Date();
  }
}