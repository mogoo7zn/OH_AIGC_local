import { hilog } from '@kit.PerformanceAnalysisKit';
//import aki from 'libentry.so';
import { Conversation } from '../model/Conversation';
import { Message } from '../model/Message';
import { HistorySidebar } from '../components/HistorySidebar';
import common from '@ohos.app.ability.common';

const DOMAIN: number = 0x0000;

@Entry
@Component
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
}