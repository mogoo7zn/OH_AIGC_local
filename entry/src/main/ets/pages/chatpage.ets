@ComponentV2
struct ChatPage {
  @Require @Param convId: string = ''  // 必须初始化
  @Local messages: Array<Message> = []  // 使用具体类型

  // 显式导入依赖
  private storage = new StorageUtil()
  private mindspore = mindsporeUtil

  build() {
    List() {
      // 组件调用语法修正
      ForEach(this.messages, (msg) => {
        ChatBubble({
          content: msg.content,
          isUser: msg.isUser
        })
      })
    }
    .width('100%')
    .height('100%')
  }
}

  private async handleSend(text: string) {
    const userMsg = new Message(text, true)
    this.messages.push(userMsg)

    // MindSpore推理
    const aiResponse = await mindspore.inference(text)
    const aiMsg = new Message(aiResponse, false)

    this.messages.push(aiMsg)
    storage.saveMessages(this.convId, this.messages)
  }
}