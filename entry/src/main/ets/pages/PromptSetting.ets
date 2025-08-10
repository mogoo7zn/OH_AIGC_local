@Entry
@Component
struct PromptSetting {
  text : string = '';
  controller: TextAreaController = new TextAreaController();

  build() {
    RelativeContainer() {
      Button(){
        Image($r('app.media.BackButton')).width(35).height(35)
      }
      .alignRules({
        center: { anchor: '__container__', align: VerticalAlign.Top },
        left: { anchor: '__container__', align: HorizontalAlign.Start },
      })
      .padding(5)
      .margin(5)
      .offset({
        y: 25,
        x: 5,
      })
      .id('back button')
      .backgroundColor(Color.White)
      .onClick(()=>{
        this.getUIContext().getRouter().back()
      })

      TextArea({
        placeholder: 'Please enter your prompt here ... \n（默认：······）',
        controller: this.controller
      })
        .alignRules({
          top: { anchor: 'back button', align: VerticalAlign.Bottom },
          left: { anchor: '__container__', align: HorizontalAlign.Start },
        })
        .offset({y:10})
        .placeholderFont({ size: 16, weight: 400 })
        .width('90%')
        .height(200)
        .padding(5)
        .margin(20)
        .border({
          radius: 8
        })
        .fontSize(16)
        .fontColor(Color.Black)
        .caretColor(Color.Black)
        .backgroundColor($r('app.color.input_background'))
        .wordBreak(WordBreak.BREAK_WORD)
        .maxLength(128)
        .showCounter(true, { thresholdPercentage: 50, highlightBorder: true })
        //计数器显示效果为用户当前输入字符数/最大字符限制数。最大字符限制数通过maxLength()接口设置。
        //如果用户当前输入字符数达到最大字符限制乘50%（thresholdPercentage）。字符计数器显示。
        //用户设置highlightBorder为false时，配置取消红色边框。不设置此参数时，默认为true。
        .enterKeyType(EnterKeyType.Send)
        .onChange((value: string) => {
          this.text = value;
        })
        .onSubmit(()=>{

        })

    }
    .height('100%')
    .width('100%')
  }
}