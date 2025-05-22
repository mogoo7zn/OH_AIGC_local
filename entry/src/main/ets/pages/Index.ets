// import { hilog } from '@kit.PerformanceAnalysisKit';
// import testNapi from 'libentry.so';
//
// const DOMAIN = 0x0000;
//
// @Entry
// @Component
// struct Index {
//   @State message: string = 'Hello World';
//
//   build() {
//     Row() {
//       Column() {
//         Text(this.message)
//           .fontSize($r('app.float.page_text_font_size'))
//           .fontWeight(FontWeight.Bold)
//           .onClick(() => {
//             this.message = 'Welcome';
//             hilog.info(DOMAIN, 'testTag', 'Test NAPI 2 + 3 = %{public}d', testNapi.add(2, 3));
//           })
//       }
//       .width('100%')
//     }
//     .height('100%')
//   }
// }

import { Chat, ChatRole, ChatMessage } from '@changwei/chatui'

@Entry
@Component
struct Index {
  build() {
    Row() {
      Column() {
        Chat({
          title:'demo chatbot',
          welcomeMessage: '我是你的测试bot',
          onSendMessage: (ctl, message) => {
            //发送用户消息
            ctl.postMessage(message)
            //显示回复等待动画
            ctl.setTyping(true)
            //3秒后发送chatbot响应消息
            setTimeout(() => {
              ctl.postMessage(new ChatMessage({
                role: ChatRole.Assistant,
                content: '这是一条测试回复'
              }))
              // 图片消息
              ctl.postMessage(new ChatMessage({
                role:ChatRole.Assistant,
                picurl:"https://foruda.gitee.com/avatar/1709712450038093632/8548349_changweizhang_1709712449.png"
              }));
            }, 3000)
          }
        })
      }
    }
    .height('100%')
  }
}