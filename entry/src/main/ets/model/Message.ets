export class Message {
  id: string = '';
  content: string = '';
  isUser: boolean = false;
  timestamp: Date = new Date();

  constructor(content: string, isUser: boolean) {
    this.id = Date.now().toString();
    this.content = content;
    this.isUser = isUser;
    this.timestamp = new Date();
  }
}