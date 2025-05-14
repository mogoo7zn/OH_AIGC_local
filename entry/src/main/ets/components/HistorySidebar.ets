import { Conversation } from '../model/Conversation';

@Component
export struct HistorySidebar {
  @Prop conversations: Array<Conversation> = [];
  onSelect: (id: string) => void = () => {};

  build() {
    Column() {
      Text('历史对话').fontSize(20).margin(10)
      List() {
        ForEach(this.conversations, (item: Conversation) => {
          ListItem() {
            Text(item.title)
              .fontSize(16)
              .padding(10)
              .width('100%')
              .onClick(() => {
                this.onSelect(item.id);
              })
          }
        }, (item: Conversation) => item.id)
      }
      .width('100%')
    }
    .width('100%')
    .height('100%')
  }
}