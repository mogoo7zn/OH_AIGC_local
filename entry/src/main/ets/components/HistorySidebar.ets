import { Conversation } from '../model/Conversation'

@ComponentV2
struct HistorySidebar {
  @Require @Param conversations: Array<Conversation> = []
  @Require @Param onSelect: (id: string) => void = () => {}

  build() {
    List() {
      ForEach(this.conversations, (conv) => {
        ListItem() {
          Row() {
            Text(conv.title)
              .fontSize(16)
              .maxLines(1)
            Text(conv.date)
              .fontColor('#666')
          }
          .onClick(() => this.onSelect(conv.id))
        }
      })
    }
    .width('100%')  // 明确宽高
    .height('100%')
  }
}