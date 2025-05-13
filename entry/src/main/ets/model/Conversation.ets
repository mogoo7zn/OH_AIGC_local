// 导入UUID生成库
import { generateUUID } from '@ohos.util'
import { Message } from '../model/Message'

export class Conversation {
  id: string = generateUUID()  // 使用标准库方法
  title: string = ''
  messages: Array<Message> = []  // 导入Message类
  date: string = new Date().toLocaleString()

  constructor(initMsg?: string) {
    if (initMsg) {
      this.title = initMsg.substring(0, 20) + '...'  // 自动生成标题
    }
  }
}