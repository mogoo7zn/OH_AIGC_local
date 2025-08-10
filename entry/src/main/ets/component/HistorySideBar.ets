// 侧边栏数据模型
class SidebarItem {
  id: string
  name: string

  constructor(id: string, name: string) {
    this.id = id
    this.name = name
  }

}

@Component
export struct SidebarComponent {
  @BuilderParam sidebarContent: () => void
  @State enterEndDeleteAreaString: string = "not enterEndDeleteArea"
  @State exitEndDeleteAreaString: string = "not exitEndDeleteArea"
  private scroller: ListScroller = new ListScroller()

  // 数据源
  // TODO : 历史对话的标题借助大模型进行主题总结
  private items: SidebarItem[] = []

  private PushSomeItem(num: number){
    for(let i=0;i<num;i++){
      this.items.push(new SidebarItem(`${i}`,`历史对话${i}`))
    }
  }

  @Builder itemEnd() {
    Row() {
      Button(){
        Image($r('app.media.ListItemDelete')).width(35).height(35)
      }
      .height('100%')
      .type(ButtonType.Circle)
      .margin("4vp")
      .backgroundColor(Color.White)


      Button(){
        Image($r('app.media.ListItemEdit')).width(35).height(35)
      }
      .height('100%')
      .type(ButtonType.Circle)
      .margin("4vp")
      .backgroundColor(Color.White)
        // .onClick(() => {
        //   this.scroller.closeAllSwipeActions()
        // })
    }
    .padding("4vp")
    .justifyContent(FlexAlign.SpaceEvenly)
  }

  aboutToAppear(): void {
    this.PushSomeItem(20);
  }

  build() {
    List({ space: 10 }) {

      /* no swipe version */

      // ForEach(this.items, (item: SidebarItem) => {
      //   ListItem() {
      //     Row() {
      //       Text(item.name)
      //         .fontSize(12)
      //         .fontColor('#333')
      //         .layoutWeight(2)
      //         .textAlign(TextAlign.Start)
      //
      //       // TODO : 每个历史对话的主题总结由此按钮驱动
      //       Button('edit')
      //         .layoutWeight(1)
      //       Button('AI')
      //         .layoutWeight(1)
      //     }
      //     .padding(12)
      //     .height(40)
      //     .width('100%')
      //     .alignItems(VerticalAlign.Top)
      //     .justifyContent(FlexAlign.Start)
      //   }
      //   .backgroundColor(Color.Gray)
      //   .borderRadius(8)
      //   .onClick(() => {
      //     console.log(`点击条目：${item.name}`)
      //   })
      // }, (item: SidebarItem) => item.id)


      /* swipe version */
      ForEach(this.items, (item: SidebarItem) => {
        ListItem() {
          Text(item.name)
            .padding(12)
            .width('100%')
            .height(50)
            .fontSize(16)
            .textAlign(TextAlign.Start)
            .borderRadius(10)
            .backgroundColor(Color.White)
        }
        .padding({
          right: 5,
          left: 5
        })
        .transition({ type: TransitionType.Delete, opacity: 0 })
        .swipeAction({
          end: {
            builder: () => { this.itemEnd() },
            onAction: () => {
              animateTo({ duration: 1000 }, () => {
                let index = this.items.indexOf(item)
                this.items.splice(index, 1)
              })
            },
            actionAreaDistance: 56,
            onEnterActionArea: () => {
              this.enterEndDeleteAreaString = "enterEndDeleteArea"
              this.exitEndDeleteAreaString = "not exitEndDeleteArea"
            },
            onExitActionArea: () => {
              this.enterEndDeleteAreaString = "not enterEndDeleteArea"
              this.exitEndDeleteAreaString = "exitEndDeleteArea"
            }
          }
        })
      }, (item: SidebarItem) => item.id)
    }
    .listDirection(Axis.Vertical) // 纵向布局[8](@ref)
    // .divider({
    //   strokeWidth: 1,
    //   color: Color.Black,
    //   startMargin: 0,
    //   endMargin: 0
    // }) // 分割线设置[9](@ref)
    .scrollBar(BarState.Auto) // 自动显示滚动条[6](@ref)
    .width('100%')
    //.border({ width: 1, color: Color.Black } )
  }
}