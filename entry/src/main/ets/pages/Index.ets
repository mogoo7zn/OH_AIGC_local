import { Chat,ChatController,IChatComponent } from '../component/Chat'
import {ChatRole} from '../common/ChatConfig'
import {ChatMessage} from '../component/ChatMessage'
import router from '@ohos.router';
import { SidebarComponent } from '../component/HistorySideBar'
import display from '@ohos.display';
import { MarginType } from '@kit.ArkUI';
import testNapi from 'libentry.so';
import { Constants as CONST} from '../common/Constants'
import { hilog } from '@kit.PerformanceAnalysisKit';

const displayClass = display.getDefaultDisplaySync();

@Entry
@Component
struct CustomInput {
  @State userInput: string = ''
  @State needBackButton: boolean = false
  @State selected_model_name : string = ''

  // 后端暴露变量
  @State text: string = '^.^'       //  模型输出
  @State modelState: number = 0     //  0：模型空置； 1：模型运行； 2：模型加载
  aboutToAppear() {

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
        Column() {
            Chat({
              title: 'demo chatbot',
              welcomeMessage: '我是你的本地AI bot',
              botMessageTextColor: Color.Black,
              userMessageTextColor: Color.Black,
              botAvatar: $r('app.media.bot_ava'), //chatbot头像
              userAvatar: $r('app.media.user_ava'),
              messageFontSize: 15,
              userInput: this.userInput,
              needInputControl:true,
              needSubmitButton: true,
              useMarkdown: true,
              onSendMessage: this.sendMessage,
              selected_model_name: this.selected_model_name,
            })
        }
        .height('100%')
        .padding({ top: '0.00vp', right: '0.00vp', bottom: '0.00vp', left: '0.00vp' })
        .backgroundColor(Color.White)
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
          //const fileName ="qwen2-0_5b-instruct-q4_0.gguf"
          const fileName = this.selected_model_name
          const fileTest = getContext().resourceDir+'/' + fileName;
          hilog.info(0, 'testTag', 'Test NAPI 2 + 3 = %{public}d', testNapi.add(2, 3));
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