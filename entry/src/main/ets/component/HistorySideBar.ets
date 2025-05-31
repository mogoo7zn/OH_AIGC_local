// 侧边栏数据模型
class SidebarItem {
  id: string
  name: string
  //icon: Resource

  constructor(id: string, name: string) {
    this.id = id
    this.name = name
    //this.icon = icon
  }
}

@Entry
@Component
export struct SidebarComponent {
  @BuilderParam sidebarContent: () => void

  // 数据源
  private items: SidebarItem[] = [
    new SidebarItem('1', '历史对话'),
    new SidebarItem('2', '历史对话'),
    new SidebarItem('3', '历史对话'),
    new SidebarItem('4', '历史对话')
  ]

  build() {
    List({ space: 10 }) {
      ForEach(this.items, (item: SidebarItem) => {
        ListItem() {
          Row() {
            // Image(item.icon)
            //   .width(24)
            //   .height(24)
            //   .margin({ right: 12 })

            Text(item.name)
              .fontSize(16)
              .fontColor('#333')
          }
          .padding(12)
          .width('100%')
          .alignItems(VerticalAlign.Top)
          .justifyContent(FlexAlign.Start)
        }
        .onClick(() => {
          console.log(`点击条目：${item.name}`)
        })
      }, (item: SidebarItem) => item.id)
    }
    .listDirection(Axis.Vertical) // 纵向布局[8](@ref)
    .divider({
      strokeWidth: 1,
      color: Color.Black,
      startMargin: 16,
      endMargin: 16
    }) // 分割线设置[9](@ref)
    .scrollBar(BarState.Auto) // 自动显示滚动条[6](@ref)
    .width(240)
    .backgroundColor('#fff')
    .border({ width: 1, color: Color.Black } )
  }
}