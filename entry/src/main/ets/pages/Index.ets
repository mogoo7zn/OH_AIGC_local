<<<<<<< Updated upstream
import { hilog } from '@kit.PerformanceAnalysisKit';
//import aki from 'libentry.so';
import { Conversation } from '../model/Conversation';
import { Message } from '../model/Message';
import { HistorySidebar } from '../components/HistorySidebar';
import common from '@ohos.app.ability.common';
=======
import { Chat, ChatRole, ChatMessage } from '@changwei/chatui'
import { ChatController } from '@changwei/chatui'
import router from '@ohos.router';
import { SidebarComponent } from '../component/HistorySideBar'
import display from '@ohos.display';
import App from '@system.app';
import TextGenerationService from '../services/TextGenerationService';
//import inferService from '../service/InferService' // 假设你有一个封装了后端推理能力的模块
>>>>>>> Stashed changes

const DOMAIN: number = 0x0000;

let CustomInputInstance: CustomInput

@Entry
@Component
<<<<<<< Updated upstream
struct MainLayout {
  @State isSidebarOpen: boolean = true;
  @State message: string = "Hello ArkUI";
  @State Prompt: string = 'x';
  @StorageLink('currentConvId') currentConvId: string = '';
  @StorageLink('conversations') conversations: Array<Conversation> = [];

  // private toggleSidebar() {
  //   animateTo({
  //     duration: 300,
  //     curve: Curve.EaseOut
  //   }, () => {
  //     this.isSidebarOpen = !this.isSidebarOpen
  //   })
  // }

  build() {
=======
struct CustomInput {
  @State userInput: string = ''
  @State needBackButton: boolean = false
  @State screenWidth: number = 0
  @State screenHeight: number = 0

  // 后端暴露变量
  @State text: string = '^.^'       //  模型输出
  @State modelState: number = 0     //  0：模型空置； 1：模型运行； 2：模型加载
  chatController = new ChatController()

  onInit() {
    CustomInputInstance = this;
  }


  aboutToAppear() {
    // 使用Display方案
    //const display = display.getDefaultDisplaySync();
    this.screenWidth = px2vp(displayClass.width);
    this.screenHeight = px2vp(displayClass.height);
  }

  build() {
    SideBarContainer(SideBarContainerType.Embed) {
      Column(){
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
              onSendMessage: (ctl, message) => {
                startInfer(message.content)
                // //发送用户消息
                // ctl.postMessage(message)
                // this.userInput = ''
                // //显示回复等待动画
                // ctl.setTyping(true)
                // //3秒后发送chatbot响应消息
                // setTimeout(() => {
                //   ctl.postMessage(new ChatMessage({
                //     role: ChatRole.Assistant,
                //     content: '这是一条测试回复    $$\alpha$$  '
                //   }))
                //   // 图片消息
                //   // ctl.postMessage(new ChatMessage({
                //   //   role:ChatRole.Assistant,
                //   //   picurl:"https://foruda.gitee.com/avatar/1709712450038093632/8548349_changweizhang_1709712449.png"
                //   // }));
                //   ctl.setTyping(false)
                //
                // }, 1000)
              }
            })
            {

              Column(){
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

                TextInput({
                  text: this.userInput
                })
                  .enterKeyType(EnterKeyType.Send)
                  .fontColor(Color.Black)
                  .backgroundColor($r('app.color.input_background'))
                    //.width('75%')
                  .flexGrow(1)
                  .onChange((val) => {
                    this.userInput = val
                  })
                  .onSubmit(() => {
                    this.chatController.submitUserInput(this.userInput)
                  })

              }

>>>>>>> Stashed changes

      SideBarContainer(SideBarContainerType.Overlay) {
        // 侧边栏内容（后述）
        if (this.isSidebarOpen) {
          HistorySidebar({
            conversations: this.conversations,
            onSelect: (id: string) => {
              this.currentConvId = id;
              this.isSidebarOpen = false;
            }
          })
            .width('30%')
            .height('100%')
            .backgroundColor(0xffffff)
        }
        // // 主内容区
        // Column() {
        //   Button($r('app.media.menu'))
        //     .onClick(() => this.isSidebarOpen = !this.isSidebarOpen)
        // }

        // 主聊天区
        Column() {
          Text(this.message)
            .fontSize(50)
            .fontWeight(FontWeight.Bold)
            // .onClick(() => {
            //   console.info(aki.SayHello("hello world")); // 调用.so文件中的代码接口
            // })
            .textAlign(TextAlign.Center)
            //.alignItems(HorizontalAlign.Center)
            .position({ y: '50%' }) // 垂直居中



          // Input区域
          TextInput({ placeholder: 'Enter the prompt' })
            .position({
              start: { value: 15, unit: 1 },
              bottom: { value: 15, unit: 1 }
            })
            .width('70%')
            .height(50)
            .borderRadius(10)
            .borderWidth(1)
            .onChange((value: string) => {
              this.Prompt = value;
            })
            .margin({ bottom: '10%' })


          Button() {
            Text('Send')
              .fontSize(25)
              .fontWeight(FontWeight.Bold)
          }
          .type(ButtonType.Capsule)
          .position({
            end: { value: 15, unit: 1 },
            bottom: { value: 15, unit: 1 }
          })
          .margin({ bottom: '10%' })
          .backgroundColor('#0D9FFB')
          .width('20%')
          .height(50)
          // .onClick(() => {
          //   this.runModel();
          // })


          Button(){
            Text('Model Type')
              .fontSize(20)
              .fontWeight(FontWeight.Bold)
          }
          .type(ButtonType.Capsule)
          .position({
            start: { value: 15, unit: 1 },
            bottom: { value: 15, unit: 1 }
          })
          .margin({ bottom: '0%' })
          .backgroundColor('#0D9FFB')
          .height(40)




          if (this.currentConvId) {
            this.ChatPage()
          } else {
            Text("新建对话").fontSize(24)
            Button("开始对话")
              .onClick(() => this.createNewConversation())
          }
        }
        .alignItems(HorizontalAlign.Center)
        .width('100%')
        //.width(this.isSidebarOpen ? '50%' : '100%')
        .height('100%')

      }
      .sideBarWidth('75%')
      .showSideBar(this.isSidebarOpen)
      .controlButton({
        left: 32,    // 距离左侧32vp
        top: 0,     // 距离顶部32vp
        width: 32,   // 按钮宽度48vp
        height: 32,  // 按钮高度48vp
        // icons: {
        //   shown: this.shownIcon,     // 展开状态图标
        //   hidden: this.hiddenIcon,   // 收起状态图标
        //   switching: this.switchingIcon // 切换过程图标
        // }
      })

  }

<<<<<<< Updated upstream
  @Builder
  ChatPage() {
    Column() {
      // 这里调用ChatPage组件的功能
      // 临时显示会话ID
      Text(`当前会话: ${this.currentConvId}`)
        .fontSize(16)
    }
  }

  private createNewConversation(): void {
    const newConv = new Conversation();
    this.conversations.push(newConv);
    this.currentConvId = newConv.id;
  }
=======
  // aboutToAppear() {
  //   const params = router.getParams(); // 获取传递过来的参数对象
  //   if(params) {
  //     this.needBackButton = params['needBackButton']
  //   }
  // }
}

/*
  开始推理
 */
function startInfer(prompt: string) {
  // 先判断模型是否已经加载
  // 注意：TextGenerationService 内部有 isModelLoaded 标记
  if (!TextGenerationService.isModelLoaded) {
    CustomInputInstance.chatController.chatCtrl?.postMessage(
      new ChatMessage({
        role: ChatRole.Assistant,
        content: '[错误] 当前尚未加载模型，请先选择模型。',
      }),
      false // 不清空输入框（这里也可以设置 true，看你需求）
    );
    return;
  }

  console.log(`[startInfer] 开始推理，prompt="${prompt}"`);
  // 1. 设置模型状态为“推理中”
  CustomInputInstance.modelState = 1;

  // 2. 先把用户消息发到界面
  CustomInputInstance.chatController.chatCtrl?.postMessage(
    new ChatMessage({
      role: ChatRole.User,
      content: prompt,
    }),
    true // 发送后清空输入框
  );

  // 3. 打开“打字中”动画（会在聊天列表里插入一个空消息，用于显示 loading）
  CustomInputInstance.chatController.chatCtrl?.setTyping(true);

  // 4. 清空本地拼接文本缓存
  CustomInputInstance.text = '';

  // 5. 调用流式推理
  TextGenerationService.generateTextStreaming(
    prompt,
    /* maxLength= */ 100,
    (token: string) => {
      // 每当收到一个 token，就追加到聊天里
      CustomInputInstance.text += token;
      CustomInputInstance.chatController.chatCtrl?.postMessage(
        new ChatMessage({
          role: ChatRole.Assistant,
          content: token,
        }),
        false // 不清空输入框，因为这是流式输出的一部分
      );
    }
  )
    .then((fullText: string) => {
      // 推理结束：关闭 loading、恢复状态
      CustomInputInstance.chatController.chatCtrl?.setTyping(false);
      CustomInputInstance.modelState = 0;
      console.log('[startInfer] 推理完成，最终生成文本：', fullText);
    })
    .catch((err: Error) => {
      // 出错：关闭 loading、恢复状态，并在聊天里提示错误
      console.error('[startInfer] 推理出错：', err);
      CustomInputInstance.chatController.chatCtrl?.setTyping(false);
      CustomInputInstance.modelState = 0;
      CustomInputInstance.chatController.chatCtrl?.postMessage(
        new ChatMessage({
          role: ChatRole.Assistant,
          content: `[推理出错] ${err.message}`,
        }),
        false
      );
    });
}

/*
  停止推理
 */
function stopInfer() {
  console.log('[stopInfer] 用户请求停止推理');
  // 如果当前并非在推理，则直接提示
  if (CustomInputInstance.modelState !== 1) {
    CustomInputInstance.chatController.chatCtrl?.postMessage(
      new ChatMessage({
        role: ChatRole.Assistant,
        content: '[提示] 当前并未在推理中。',
      }),
      false
    );
    return;
  }

  // 取消推理：立刻把状态变为 0，关闭 loading
  CustomInputInstance.modelState = 0;
  CustomInputInstance.chatController.chatCtrl?.setTyping(false);
  CustomInputInstance.chatController.chatCtrl?.postMessage(
    new ChatMessage({
      role: ChatRole.Assistant,
      content: '[已取消推理]',
    }),
    false
  );

  // 如果你后续给 TextGenerationService 增加了 cancel 功能，可以在这里调用：
  // (TextGenerationService as any).cancelInference();
}

/*
  结束模型使用
 */
function dropModel() {
  console.log('[dropModel] 用户请求释放模型资源');
  // 1. 如果当前正在推理，就先“打断”推理
  if (CustomInputInstance.modelState === 1) {
    CustomInputInstance.chatController.chatCtrl?.postMessage(
      new ChatMessage({
        role: ChatRole.Assistant,
        content: '[提示] 推理已中断，正在释放模型...',
      }),
      false
    );
    CustomInputInstance.modelState = 0;
    CustomInputInstance.chatController.chatCtrl?.setTyping(false);
    // 同样，如果实现了真正的 cancelInference 可以在这里调用：
    // (TextGenerationService as any).cancelInference();
  }

  // 2. “卸载”模型：当前 TextGenerationService 并未提供 unload 接口，
  //    所以这里只有把 isModelLoaded 手动设为 false。
  TextGenerationService.isModelLoaded = false;

  // 3. 清空本地缓存文本（可选）
  CustomInputInstance.text = '';

  // 4. 在聊天里提示用户：
  CustomInputInstance.chatController.chatCtrl?.postMessage(
    new ChatMessage({
      role: ChatRole.Assistant,
      content: '[模型已释放]',
    }),
    false
  );
}

/*
  选择模型
 */
function chooseMeodel(name: string) {
  // console.log(`[chooseModel] 用户选择模型：${name}`);
  //
  // // 1. 进入“加载中”状态
  // CustomInputInstance.modelState = 2;
  // CustomInputInstance.chatController.chatCtrl?.postMessage(
  //   new ChatMessage({
  //     role: ChatRole.Assistant,
  //     content: `[加载模型：${name} 中...]`,
  //   }),
  //   false
  // );
  //
  // // 2. 构造模型文件与词表路径（请根据实际项目路径修改）
  // const modelPath = `/data/app/models/${name}.mnn`;
  // const vocabPath = `/data/app/models/${name}.vocab.txt`;
  //
  // // 3. 调用 TextGenerationService.loadModel
  // const ok = await TextGenerationService.loadModel(modelPath, vocabPath);
  // if (ok) {
  //   console.log(`[chooseModel] 模型 ${name} 加载成功`);
  //   CustomInputInstance.chatController.chatCtrl?.postMessage(
  //     new ChatMessage({
  //       role: ChatRole.Assistant,
  //       content: `[已加载模型：${name}]`,
  //     }),
  //     false
  //   );
  // } else {
  //   console.error(`[chooseModel] 模型 ${name} 加载失败`);
  //   CustomInputInstance.chatController.chatCtrl?.postMessage(
  //     new ChatMessage({
  //       role: ChatRole.Assistant,
  //       content: `[模型加载失败：${name}]`,
  //     }),
  //     false
  //   );
  // }
  //
  // // 4. 恢复状态为“空闲”
  // CustomInputInstance.modelState = 0;
>>>>>>> Stashed changes
}