import { Chat,ChatController,IChatComponent } from '../component/Chat'
import {ChatRole} from '../common/ChatConfig'
import {ChatMessage} from '../component/ChatMessage'
import router from '@ohos.router';
import { SidebarComponent } from '../component/HistorySideBar'
import display from '@ohos.display';
import App from '@system.app';
import { MarginType } from '@kit.ArkUI';
import testNapi from 'libentry.so';
import { Constants as CONST} from '../common/Constants'

const displayClass = display.getDefaultDisplaySync();

@Entry
@Component
struct CustomInput {
  @State userInput: string = ''
  //static userInputPropmt: string = '在此处输入...'
  @State needBackButton: boolean = false
  chatController = new ChatController()
  @State screenWidth: number = 0
  @State screenHeight: number = 0

  // 后端暴露变量
  @State text: string = '^.^'       //  模型输出
  @State modelState: number = 0     //  0：模型空置； 1：模型运行； 2：模型加载
  aboutToAppear() {
    // 使用Display方案
    //const display = display.getDefaultDisplaySync();
    this.screenWidth = px2vp(displayClass.width);
    this.screenHeight = px2vp(displayClass.height);
  }

  build() {
    SideBarContainer(SideBarContainerType.Embed) {
      //悬浮窗视图
      Column(){
        Text('历史对话')
          .fontSize(25)//.textAlign(Alignment.Center)
          .margin({top: 12, bottom:12})
          .height(35)

        SidebarComponent(){
        }
      }
      .backgroundColor($r('app.color.sidebar_background'))
      //内容区视图
      Column(){
        Column() {
            Chat({
              title: 'demo chatbot',
              needTitleBar: true,
              welcomeMessage: '我是你的本地AI bot',
              botMessageBackgroundColor: $r('app.color.light_blue_bg'),
              botMessageTextColor: Color.Black,
              userMessageBackgroundColor: Color.White,
              userMessageTextColor: Color.Black,
              botAvatar: $r('app.media.bot_ava'), //chatbot头像
              userAvatar: $r('app.media.user_ava'),
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
              onSendMessage: this.sendMessage,
            })
            {
              Column(){
                Row() {
                  Button() {
                    Image($r('app.media.model_choose')).width(25).height(25)
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
                    Image($r('app.media.add_more')).width(25).height(25)
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
                .layoutWeight(1)
                //.margin(5)

                Row(){
                  TextArea({
                    placeholder: CONST.InputPropmt,
                  })
                    .enterKeyType(EnterKeyType.Send)
                    .fontColor(Color.Black)
                    .backgroundColor($r('app.color.input_background'))
                    .width('75%')
                    .flexGrow(1)
                    .onChange((val) => {
                      this.userInput = val
                    })
                    .onSubmit(() => {
                      this.chatController.submitUserInput(this.userInput)
                      this.userInput = ''
                    })
                    // .cancelButton({
                    //   style: CancelButtonStyle.INPUT,
                    //   icon: {
                    //     size: 20,
                    //     src: $r('app.media.InputDelete'),
                    //     color: Color.Gray
                    //   }
                    // })
                    .caretColor(Color.Black)
                    .fontSize(12)
                    .wordBreak(WordBreak.BREAK_ALL)
                    .height('100%')
                    .margin({ top: 0, right: 5, bottom: 0, left: 5 })
                    //.padding({ top: 5, right: 5, bottom: 0, left: 5 })

                  Button(){
                    Image($r('app.media.SendButton')).width(30).height(30)
                  }
                  .height('100%')
                  .type(ButtonType.Circle)
                  .backgroundColor($r('app.color.sendbutton_background'))
                  .onClick(() => {
                    this.chatController.submitUserInput(this.userInput)
                    this.userInput = ''
                  })
                }
                .layoutWeight(1)
                .width('100%')

              }
            }
        }
        .height('100%')
        .padding({ top: '0.00vp', right: '0.00vp', bottom: '0.00vp', left: '0.00vp' })
      }
    }
    .showControlButton(true)
    .controlButton({
      left: 10,
      top: 15,
      width: 30,
      height: 30,
      icons: {
        shown: $r("app.media.SideBarButton2") ,
        hidden: $r("app.media.SideBarButton2") ,
        switching: $r("app.media.SideBarButton2")
      }
    })
    .sideBarWidth('75%')
  }

  private async sendMessage(ctl:IChatComponent,message:ChatMessage){
      //发送用户消息
      ctl.postMessage(message)
      this.userInput = ''
      //显示回复等待动画
      ctl.setTyping(true)
      //3秒后发送chatbot响应消息
      try {
          const fileName ="qwen2-0_5b-instruct-q4_0.gguf"
          const fileTest = getContext().resourceDir+'/' + fileName;
          testNapi.load_module(fileTest);
          testNapi.inference_start(message.content,
            (token:string)=>{
                ctl.deleteMessage()
                ctl.postMessage(new ChatMessage({
                  role: ChatRole.Assistant,
                  content: token
                }))
            })
      } catch (err){
        console.error("test:error,%s",err);
      }
  }
  // aboutToAppear() {
  //   const params = router.getParams(); // 获取传递过来的参数对象
  //   if(params) {
  //     this.needBackButton = params['needBackButton']
  //   }
  // }
  //private onSendMessage(ctl:IChatComponent, message){

  //}
}

/*
  开始推理
 */
function startInfer(prompt: string) {

}

/*
  停止推理
 */
function stopInfer() {

}

/*
  结束模型使用
 */
function dropModel() {

}

/*
  选择模型
 */
function chooseMeodel(name: string) {

}