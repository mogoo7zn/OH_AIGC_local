import { hilog } from '@kit.PerformanceAnalysisKit';
import aki from 'libentry.so';
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
  @StorageLink('currentConvId') currentConvId: string = '';
  @StorageLink('conversations') conversations: Array<Conversation> = [];

  build() {
    Column() {
      Row() {
        // 侧边栏历史列表
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
        }

        // 主聊天区
        Column() {
          Text(this.message)
            .fontSize(50)
            .fontWeight(FontWeight.Bold)
            .onClick(() => {
              console.info(aki.SayHello("hello world")); // 调用.so文件中的代码接口
            })

          if (this.currentConvId) {
            this.ChatPage()
          } else {
            Text("新建对话").fontSize(24)
            Button("开始对话")
              .onClick(() => this.createNewConversation())
          }
        }
        .width(this.isSidebarOpen ? '70%' : '100%')
        .height('100%')
      }
      .width('100%')
      .height('100%')

      // 底部菜单按钮
      Button('菜单')
        .position({x: 0, y: '90%'})
        .onClick(() => {
          this.isSidebarOpen = !this.isSidebarOpen;
        })
    }
    .width('100%')
    .height('100%')
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