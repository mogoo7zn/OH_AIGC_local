import { preferences } from '@kit.ArkData';
import { util } from '@kit.ArkTS';
import { BusinessError } from '@kit.BasicServicesKit';
import { Constants as CONST} from '../common/Constants'


let dataPreferences: preferences.Preferences ;

@Entry
@Component
struct PromptSetting {
  @State prompt : string = '你是一个智能助手。\n'
    +
    '\n'
    +
    '正常回答用户问题。\n'
    +
    '\n'
    +
    '如果用户表达了 打开某个应用的意图，无论是否直接提到应用名称（如“打开微信”、“我要听音乐”、“帮我查地图”）：\n'
    +
    '\n'
    +
    '先简短地回复用户。\n'
    +
    '\n'
    +
    '然后额外输出一个 JSON 对象，格式如下：\n'
    +
    '\n'
    +
    '{ "open": "应用名", "url": "应用的真实URL" }\n'
    +
    '\n'
    +
    '\n'
    +
    '"应用名" 使用常见、自然的中文名称（如“微信”、“支付宝”、“地图”、“音乐”、“浏览器”）。\n'
    +
    '\n'
    +
    '"url" 替换为该应用的真实 URL（不要使用示例 URL，不要留空）。\n'
    +
    '\n'
    +
    '如果用户没有这种意图，就不要输出该字段。\n'
    +
    '\n'
    +
    '严格保证 JSON 格式正确，不要多余解释或附加文本。\n'
    +
    '\n'
    +
    '模糊场景示例：\n'
    +
    '\n'
    +
    '用户说 “帮我查地图” → 输出 { "open": "地图", "url": "https://maps.gaode.com" }\n'
    +
    '\n'
    +
    '用户说 “我要听音乐” → 输出 { "open": "音乐", "url": "https://music.qq.com" }'
  controller: TextAreaController = new TextAreaController();

  aboutToAppear(): void {
    let context = this.getUIContext().getHostContext() as Context
    let options: preferences.Options = { name: 'myStore' };
    dataPreferences = preferences.getPreferencesSync(context, options);
  }

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

      Button('保存')
        .fontSize(16)
        .fontColor(Color.Black)
        .backgroundColor(Color.White)
        .height(35)
        .alignRules({
          center: { anchor: '__container__', align: VerticalAlign.Top },
          right: { anchor: '__container__', align: HorizontalAlign.End },
        })
        .padding(5)
        .margin(5)
        .offset({
          y: 25,
          x: -5,
        })
        .onClick(()=>{
          this.promptWrite(this.prompt);
          this.getUIContext().showAlertDialog({
            // title: '提示',
            message: '您的prompt已保存！',
            autoCancel: true,
            alignment: DialogAlignment.Center,

            offset: { dx: 0, dy: 0 },
            gridCount: 3,
            width: 300,
            height: 125,
            backgroundColor: Color.White,
            textStyle: { wordBreak: WordBreak.BREAK_ALL },
            confirm: {
              value: '好的',
              action: () => {
                console.info('Button-clicking callback');
              }
            },
            cancel: () => {
              console.info('Closed callbacks');
            },
            onWillDismiss: (dismissDialogAction: DismissDialogAction) => {
              console.info(`reason= ${dismissDialogAction.reason}`);
              console.info('AlertDialog onWillDismiss');
              if (dismissDialogAction.reason === DismissReason.PRESS_BACK) {
                dismissDialogAction.dismiss();
              }
              if (dismissDialogAction.reason === DismissReason.TOUCH_OUTSIDE) {
                dismissDialogAction.dismiss();
              }
            }
          })
        })

      TextArea({
        placeholder: 'Please enter your prompt here ... \n（默认：······）',
        controller: this.controller,
        text: this.prompt
      })
        .alignRules({
          top: { anchor: 'back button', align: VerticalAlign.Bottom },
          left: { anchor: '__container__', align: HorizontalAlign.Start },
        })
        .offset({y:10})
        .placeholderFont({ size: 16, weight: 400 })
        .width('90%')
        .height(400)
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
        .maxLength(512)
        .showCounter(true, { thresholdPercentage: 50, highlightBorder: true })
        //计数器显示效果为用户当前输入字符数/最大字符限制数。最大字符限制数通过maxLength()接口设置。
        //如果用户当前输入字符数达到最大字符限制乘50%（thresholdPercentage）。字符计数器显示。
        //用户设置highlightBorder为false时，配置取消红色边框。不设置此参数时，默认为true。
        .enterKeyType(EnterKeyType.Send)
        .onChange((value: string) => {
          this.prompt = value;
        })
        .onSubmit(()=>{
          this.promptWrite(this.prompt);
        })
        .onAppear(()=>{
          let getprompt = this.promptGet()
          if(getprompt != 'default'){
            this.prompt = getprompt
          }
        })
    }
    .height('100%')
    .width('100%')
  }

  promptWrite (prompt : string) {
    CONST.prompt = prompt;
    dataPreferences.putSync('prompt', prompt);
    dataPreferences.flush((err: BusinessError) => {
      if (err) {
        console.error(`Failed to flush. Code:${err.code}, message:${err.message}`);
        return;
      }
      console.info('Succeeded in flushing.');
    })
  }

  promptGet () : string {
    let val = dataPreferences.getSync('prompt', 'default') as string;
    return val;
  }
}