import { Chat, ChatRole, ChatMessage } from '@changwei/chatui'
import { ChatController } from '@changwei/chatui'
import router from '@ohos.router';
import { SidebarComponent } from '../component/HistorySideBar'
import display from '@ohos.display';
const displayClass = display.getDefaultDisplaySync();

@Entry
@Component
struct CustomInput {
  @State userInput: string = ''
  @State needBackButton: boolean = false
  chatController = new ChatController()
  @State screenWidth: number = 0
  @State screenHeight: number = 0

  aboutToAppear() {
    // 使用Display方案
    //const display = display.getDefaultDisplaySync();
    this.screenWidth = px2vp(displayClass.width);
    this.screenHeight = px2vp(displayClass.height);
  }

  build() {
    SideBarContainer(SideBarContainerType.Embed) {
      Column(){
        Text('历史对话')//.textAlign(Alignment.Center)
        SidebarComponent(){

        }
      }

      Column(){
        Row() {
          Column() {
            Chat({
              title: 'demo chatbot',
              needTitleBar: true,
              welcomeMessage: '我是你的本地AI bot',
              botMessageBackgroundColor: Color.Grey,
              botMessageTextColor: Color.Black,
              userMessageBackgroundColor: Color.White,
              userMessageTextColor: Color.Black,
              botAvatar: $r('app.media.bot'), //chatbot头像
              userAvatar: $r('app.media.human'),
              messageFontSize: 15,
              userInput: this.userInput,
              controller: this.chatController,
              needBackButton: this.needBackButton, //是否显示顶部返回按钮。点击返回导航上一页
              needInputControl:true,
              needSubmitButton: true,
              submitButtonText: "send",
              inputTextPlaceHolder: "给你的本地AI发送消息",
              inputTextPlaceHolderColor: Color.Black,
              inputTextColor: Color.Black,
              useMarkdown: true,
              onSendMessage: (ctl, message) => {
                //发送用户消息
                ctl.postMessage(message)
                this.userInput = ''
                //显示回复等待动画
                ctl.setTyping(true)
                //3秒后发送chatbot响应消息
                setTimeout(() => {
                  ctl.postMessage(new ChatMessage({
                    role: ChatRole.Assistant,
                    content: '这是一条测试回复这是一条测试回复这是一条测试回复这是一条测试回复这是一条测试回复这是一条测试回复'
                  }))
                  // 图片消息
                  ctl.postMessage(new ChatMessage({
                    role:ChatRole.Assistant,
                    picurl:"https://foruda.gitee.com/avatar/1709712450038093632/8548349_changweizhang_1709712449.png"
                  }));

                }, 3000)
              }
            }) {

              Column(){

                TextInput({
                  text: this.userInput
                })
                  .enterKeyType(EnterKeyType.Send)
                  .fontColor(Color.Black)
                  .backgroundColor(Color.White)
                    //.width('75%')
                  .flexGrow(1)
                  .onChange((val) => {
                    this.userInput = val
                  })
                  .onSubmit((ss) => {
                    this.chatController.submitUserInput(this.userInput)
                  })

                Row() {
                  Button() {
                    Image($r('app.media.model_choose'))
                  }
                  .backgroundColor(Color.White)
                  .aspectRatio(1)        // 宽高比1:1 → 高度=宽度
                  .width('30')
                  .margin(5)
                  .onClick(() => {
                    ActionSheet.show({
                      title: '您的模型',
                      message: '请选择所需模型',
                      sheets: [
                        { title: 'Qwen3', action: () => {
                          console.log('Get Alert Dialog handled')
                        } },
                        { title: 'Stable Diffusion', action: () => {
                          console.log('Get Alert Dialog handled')
                        } }
                      ],
                      confirm: {
                        value: '取消',
                        action: () => console.log('已取消')
                      }
                    })

                  })

                  Button(){
                    Image($r('app.media.add_more'))
                  }
                  .backgroundColor(Color.White)
                  .aspectRatio(1)        // 宽高比1:1 → 高度=宽度
                  .width('30')
                  .margin(5)
                  .onClick(() => {
                    ActionSheet.show({
                      title: '选择您要上传的文件类型',
                      message: '',
                      sheets: [
                        { title: '图片', action: () => { } },
                        { title: '音频', action: () => {} }
                      ],
                      confirm: {
                        value: '取消',
                        action: () => console.log('已取消')
                      }
                    })

                  })




                }
                .alignItems(VerticalAlign.Center) // 垂直居中对齐
                .width('100%')


              }



            }

          }
        }
        .height('100%')
      }

    }
  }

  // aboutToAppear() {
  //   const params = router.getParams(); // 获取传递过来的参数对象
  //   if(params) {
  //     this.needBackButton = params['needBackButton']
  //   }
  // }
}