import { hilog } from '@kit.PerformanceAnalysisKit';
import testNapi from 'libentry.so';
import aki from 'libentry.so'

const DOMAIN = 0x0000;

@Entry
@Component
struct MainLayout {
  @State isSidebarOpen: boolean = true
  @StorageLink('currentConvId') currentConvId: string = ''
  @StorageLink('conversations') conversations: Conversation[] = []

  build() {
    SideBarContainer(SideBarContainerType.Embed) {
      // 侧边栏历史列表
      HistorySidebar({
        conversations: this.conversations,
        onSelect: (id: string) => {
          this.currentConvId = id
          this.isSidebarOpen = false
        }
      })

      // 主聊天区
      Column() {
        Text(this.message)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            console.info(aki.SayHello("hello world")); // 调用.so文件中的代码接口
          })

        if (this.currentConvId) {
          ChatPage({ convId: this.currentConvId })
        } else {
          Text("新建对话").fontSize(24)
          Button("开始对话")
            .onClick(() => this.createNewConversation())
        }
      }
    }
    .sideBarWidth('30%')
    .controlButton({ icon: $r('app.media.menu'), position: 'start' })
  }

  private createNewConversation() {
    const newConv = new Conversation()
    this.conversations.push(newConv)
    this.currentConvId = newConv.id
  }
}