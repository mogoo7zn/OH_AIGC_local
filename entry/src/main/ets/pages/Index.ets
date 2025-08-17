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
import { fileIo } from '@kit.CoreFileKit';
import { preferences } from '@kit.ArkData';

import ChatHistoryTable from '../common/database/tables/ChatHistoryTable';
import ChatHistoryData from '../viewmodel/ChatHistoryData';
import SessionTable from '../common/database/tables/SessionTable';
import SessionData from '../viewmodel/SessionData';
import { common, Want, OpenLinkOptions } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';

const displayClass = display.getDefaultDisplaySync();
let dataPreferences: preferences.Preferences ;

const TAG: string = '[UIAbilityComponentsOpenLink]';
const DOMAIN_NUMBER: number = 0xFF00;

@Entry
@Component
struct CustomInput {
  @State userInput: string = ''
  @State needBackButton: boolean = false
  @State selected_model_name : string = ''
  prompt : string = ''
  last_token = '';
  //@Consume('modeluri') model_uri : string

  @State currentSessionId: number = -1;   // 当前会话ID
  @State conversationCount: number = 0;   // 当前会话中的对话数量

  private chatHistoryTable: ChatHistoryTable = new ChatHistoryTable();
  private sessionTable: SessionTable = new SessionTable();

  // 后端暴露变量
  @State text: string = '^.^'       //  模型输出
  @State modelState: number = 0     //  0：模型空置； 1：模型运行； 2：模型加载
  aboutToAppear() {
    let context = this.getUIContext().getHostContext() as Context
    let options: preferences.Options = { name: 'myStore' };
    dataPreferences = preferences.getPreferencesSync(context, options);
    // 初始化数据库
    this.chatHistoryTable.getRdbStore(() => {
      console.info('SessionTable 初始化成功');
    })
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
              title: 'OpenHarmony AI',
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
    this.prompt = dataPreferences.getSync('prompt', 'default') as string
      try {
          //const fileName ="qwen2-0_5b-instruct-q4_0.gguf"
          const modeluri = this.selected_model_name

        const model = CONST.model_uri
        let file = fileIo.openSync(CONST.model_uri, fileIo.OpenMode.READ_ONLY);

        const multimodeluri0 = 'file://docs/storage/Users/currentUser/Download/model.gguf'
        const multimodeluri1 = 'file://docs/storage/Users/currentUser/Download/mmproj.gguf'

          const fileTest = "/storage/Users/currentUser/Download"+'/';

          if(!message.picurl){

            testNapi.load_module(file.path,this.prompt);
            testNapi.inference_start(message.content,
              (token:string)=>{
                //todo
                if(token == '炸薯丸出品'){
                  // todo ：存入数据库

                  // 分析意图
                  analyzeIntension(this.last_token,this.getUIContext().getHostContext() as common.UIAbilityContext)
                  return
                }
                ctl.deleteMessage()
                ctl.postMessage(new ChatMessage({
                  role: ChatRole.Assistant,
                  content: token
                }))
                this.last_token = token;
              })
          } else {
            testNapi.load_multimodal_module(multimodeluri0,multimodeluri1)
            testNapi.load_multimodal_image(message.picurl)
            testNapi.inference_multimodal_start(message.content,
              (token:string)=>{
                ctl.deleteMessage()
                ctl.postMessage(new ChatMessage({
                  role: ChatRole.Assistant,
                  content: token
                }))
                this.last_token = token;
              })
          }

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

// function pickAPP(){
//   let context = this.getUIContext().getHostContext() as common.UIAbilityContext;
//   let wantParam: Record<string, Object> = {
//     'sceneType': 3,
//     'destinationName': '武汉市华中科技大学',
//   };
//   let abilityStartCallback: common.AbilityStartCallback = {
//     onError: (code: number, name: string, message: string) => {
//       console.log(`onError code ${code} name: ${name} message: ${message}`);
//     }
//   }
//
//   context.startAbilityByType("navigation", wantParam, abilityStartCallback,
//     (err) => {
//       if (err) {
//         console.error(`startAbilityByType fail, err: ${JSON.stringify(err)}`);
//       } else {
//         console.log(`success`);
//       }
//     });
// }

function openlink(url: string,context: common.UIAbilityContext){
  let link: string = url;
  let openLinkOptions: OpenLinkOptions = {
    appLinkingOnly: false
  };

  try {
    context.openLink(link, openLinkOptions)
      .then(() => {
        hilog.info(DOMAIN_NUMBER, TAG, 'openLink success.');
      }).catch((err: BusinessError) => {
      hilog.error(DOMAIN_NUMBER, TAG, `openLink failed. Code is ${err.code}, message is ${err.message}`);
    });
  } catch (paramError) {
    hilog.error(DOMAIN_NUMBER, TAG, `Failed to start link. Code is ${paramError.code}, message is ${paramError.message}`);
  }
}

function  analyzeIntension(token: string,ctx: common.UIAbilityContext){
  let url = parseJsonFromEnd(token).url
  if(url != ''){
    openlink(url,ctx);
  }
}

// 定义 JSON 对象类型
interface AppConfig {
  open: string;
  url: string;
}

// 主函数：从字符串末尾解析 JSON
function parseJsonFromEnd(input: string): AppConfig {
  let nolink : AppConfig = {open:'',url:''}
  // 1. 检查输入是否有效
  if (!input || input.length === 0) {
    console.error("输入字符串为空");
    return nolink;
  }

  // 2. 从字符串末尾开始查找可能的 JSON 起始位置
  let jsonStartIndex = -1;
  let braceCount = 0;
  let inString = false;
  let escapeNext = false;

  // 从后向前扫描字符串
  for (let i = input.length - 1; i >= 0; i--) {
    const char = input[i];

    // 处理转义字符
    if (char === '\\' && !escapeNext) {
      escapeNext = true;
      continue;
    }

    // 处理字符串边界
    if (char === '"' && !escapeNext) {
      inString = !inString;
    }

    // 只在非字符串区域处理大括号
    if (!inString) {
      if (char === '}') {
        braceCount++;
      } else if (char === '{') {
        braceCount--;

        // 当找到匹配的起始大括号时
        if (braceCount === 0) {
          jsonStartIndex = i;
          break;
        }
      }
    }

    escapeNext = false;
  }

  // 3. 验证是否找到有效的 JSON 对象
  if (jsonStartIndex === -1) {
    console.error("未找到有效的 JSON 对象");
    return nolink;
  }

  // 4. 提取 JSON 字符串片段
  const jsonString = input.substring(jsonStartIndex);

  try {
    // 5. 尝试解析 JSON
    const parsed = JSON.parse(jsonString) as AppConfig;

    // 6. 验证必需字段
    if (typeof parsed.open === 'string' && typeof parsed.url === 'string') {
      return parsed;
    } else {
      console.error("JSON 对象缺少必需的 open 或 uri 字段");
      return nolink;
    }
  } catch (error) {
    console.error("JSON 解析失败:", error);
    return nolink;
  }
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