import { ChatMessage } from './ChatMessage'
import { ChatDataSource, IChatDataSource } from './ChatDataSource'
import inputMethod from '@ohos.inputMethod';
import { ChatConfig, ChatRole } from '../common/ChatConfig'
import font from '@ohos.font';
import router from '@ohos.router';
import { lvMarkdownIn, lvTitle, lvText, lvLink, lvCode, lvQuote,  } from '@luvi/lv-markdown-in'
import { Constants as CONST} from '../common/Constants'
import { UIContext } from '@kit.ArkUI';
import { lvImage } from '@luvi/lv-markdown-in/src/main/ets/lv-markdown-in/lib/domain/LvImage';

//设置各级标题字号
lvTitle.setLevel1Title(18)
lvTitle.setLevel2Title(16)
lvTitle.setLevel3Title(14)
lvTitle.setLevel4Title(12)
lvTitle.setLevel5Title(10)
lvTitle.setLevel6Title(8)

lvText.setTextSize(8)                   // 设置文本字号20
lvText.setTextColor("red")               // 设置文本颜色为红色
lvText.setTextMarkBackground("#F7FF03")     // 设置标记文本的背景颜色为黄色

lvLink.setTextSize(8)            // 设置超链接文本字号20
lvLink.setTextColor("red")        // 设置超链接文本颜色为红色
lvLink.setTextUnderline(true)     // 设置超链接下划线

lvImage.setImgWidth("70%")         // 设置图片宽度为70%
lvImage.setImgHeight(null)         // 设置图片高度为null
lvImage.setConfGlobal(true)        // 强制开启图片全局宽高

lvCode.setTheme("dark")         // 设置代码块主题为暗夜

lvQuote.setBackgroundColor("rgba(234, 239, 255, 0.62)")      // 设置引用块背景颜色为淡粉色
lvQuote.setBorderColor("red")                                // 设置引用块左边颜色为红色

interface IChatComponent {
  userInput: string

  //发送消息到UI
  //clearInput：指示是否清空输入框
  postMessage(msg: ChatMessage, clearInput?: boolean): void

  deleteMessage() : void

  setTyping(isTyping: boolean): void

  submitUserInput(userIputText: string): void

  changeMessage(msg:ChatMessage):void
}

class ChatController {
  chatCtrl?: IChatComponent
  // userInput: string = '';
  // messages: IChatDataSource = new ChatDataSource();
  // listScroller: Scroller = new Scroller();
  // isResponding: boolean = false;
  // onSendMessage?: (ctrl: IChatComponent, msg: ChatMessage) => void;
  //
  //
  //   postMessage(msg: ChatMessage, clearInput: boolean = true) {
  //     let lastMsg: ChatMessage = this.messages.getData(this.messages.totalCount() - 1)
  //     if (!lastMsg.content) {
  //       this.messages.updateData(this.messages.totalCount() - 1, msg)
  //     } else {
  //       this.messages.pushData(msg)
  //     }
  //     if (clearInput) {
  //       this.userInput = ''
  //     }
  //     this.listScroller.scrollToIndex(this.messages.totalCount() - 1)
  //     this.isResponding = false
  //   }
  //   deleteMessage(){
  //     this.messages.removeData(this.messages.totalCount()-1)
  //   }
  //   setTyping(isTyping: boolean = false) {
  //     if (isTyping) {
  //       this.messages.pushData(new ChatMessage({
  //         role: ChatRole.Assistant,
  //         content: ""
  //       }))
  //       this.listScroller.scrollToIndex(this.messages.totalCount() - 1)
  //     } else {
  //       let lastMsg: ChatMessage = this.messages.getData(this.messages.totalCount() - 1)
  //       if (!lastMsg.content) {
  //         this.messages.removeData(this.messages.totalCount() - 1)
  //       }
  //     }
  //     this.isResponding = isTyping
  //   }
  //   submitUserInput(userIputText: string) {
  //     this.userInput = userIputText
  //     if (this.userInput && !this.isResponding) {
  //       if (this.onSendMessage) {
  //         this.isResponding = true
  //         this.onSendMessage(this, new ChatMessage({
  //           role: ChatRole.User,
  //           content: this.userInput
  //         }))
  //       }
  //     }
  //   }
  //   changeMessage(msg:ChatMessage){
  //     console.info("test:msg=%s,c=%s",msg.content,this.messages.getString(this.messages.totalCount()-1));
  //     this.messages.updateData(this.messages.totalCount()-1 , msg)
  //     console.info("test:msg=%s",msg.content);
  //     this.listScroller.scrollToIndex(this.messages.totalCount() - 1)
  //     this.userInput = ''
  //     this.isResponding = false
  //   }

}


//@Preview
@Component
struct Chat {

  controller = new ChatController()
  Textcontroller: TextAreaController = new TextAreaController()

  private selected_model_index: number | number[] = 0;
  private models: string[] = ['qwen2.5-3b-instruct-q8_0.gguf', 'qwen2-0_5b-instruct-q4_0.gguf', 'qwen2.5-7b-instruct-q3_k_m.gguf', 'DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf'];
  @Prop selected_model_name : string = '';
  @State hasSelect : boolean = false;

  @State messages: IChatDataSource = new ChatDataSource();
  @State userInput: string = ''
  @State isResponding: boolean = false
  private listScroller: Scroller = new Scroller();
  public onClear?: (event?: ClickEvent) => void;
  public onSendMessage?: (ctl: IChatComponent, message: ChatMessage) => void;
  public botAvatar: Resource = $r('app.media.chatbot')
  public userAvatar: Resource = $r('app.media.me')
  public backIcon: Resource = $r('app.media.back')
  public clearChatIcon: Resource = $r('app.media.clear')
  title: string = "ChatBot"
  public welcomeMessage: string = "主人好，我是智能助理，你的贴心小助手~"
  //自定义消息颜色
  //public botMessageBackgroundColor: ResourceColor = '#fff'
  public botMessageTextColor: ResourceColor = Color.Black
  // public userMessageBackgroundColor: ResourceColor = '#96ed6a'
  public userMessageTextColor: ResourceColor = Color.Black
  public messageFontSize: number = 18
  public needSubmitButton: boolean = true
  //public submitButtonText: string = "Send"
  //public inputTextColor: ResourceColor = Color.Black
  @State needInputControl: boolean = true
  //@State needBackButton: boolean = false
  // public controller?: ChatController
  public useMarkdown: boolean = true

  submitUserInput(userIputText: string) {
    this.userInput = userIputText
    if (this.userInput && !this.isResponding) {
      if (this.onSendMessage) {
        this.isResponding = true
        this.onSendMessage(this, new ChatMessage({
          role: ChatRole.User,
          content: this.userInput
        }))
      }
    }
  }

  // @BuilderParam InputControl: () => void = this.defaultInputText
  //
  // @Builder
  // defaultInputText() {
  //   TextInput({
  //     placeholder: this.inputTextPlaceHolder,
  //     text: this.userInput
  //   })
  //     .fontColor(this.inputTextColor)
  //     .placeholderColor(this.inputTextPlaceHolderColor)
  //     .layoutWeight(1)
  //     .onChange((value: string) => {
  //       console.info(value);
  //       this.userInput = value;
  //     })
  //     // .onSubmit((EnterKeyType) => {
  //     //   this.submitUserInput(this.userInput)
  //     // })
  //
  //   if (this.needSubmitButton) {
  //     Button(this.submitButtonText)
  //       .backgroundColor(0x317aff)
  //       .margin({ left: 10 })
  //       .width(60)
  //       .height(30)
  //       .onClick(() => {
  //         this.hideKeyboard()
  //         this.submitUserInput(this.userInput)
  //       })
  //   }
  // }

  public setDefaultMessage(role: ChatRole, content: string) {
    if (content) {
      let systemMsg = new ChatMessage({
        role: role,
        content: content
      })
      this.messages.clearData();
      this.messages.pushData(systemMsg);
    } else {
      this.messages.clearData();
    }
  }

  //API 9接口，需要系统应用权限
  //API 10使用hideTextInput
  hideKeyboard() {
    let inputMethodController = inputMethod.getController();
    //inputMethodController.hideSoftKeyboard()
    inputMethodController.hideTextInput();
  }

  //@State isTyping: boolean = false;

  setTyping(isTyping: boolean = false) {
    if (isTyping) {
      this.messages.pushData(new ChatMessage({
        role: ChatRole.Assistant,
        content: ""
      }))
      this.listScroller.scrollToIndex(this.messages.totalCount() - 1)
    } else {
      let lastMsg: ChatMessage = this.messages.getData(this.messages.totalCount() - 1)
      if (!lastMsg.content) {
        this.messages.removeData(this.messages.totalCount() - 1)
      }
    }
    this.isResponding = isTyping
  }

  deleteMessage(){
    this.messages.removeData(this.messages.totalCount()-1)
  }

  postMessage(msg: ChatMessage, clearInput: boolean = true) {
    let lastMsg: ChatMessage = this.messages.getData(this.messages.totalCount() - 1)
    if (!lastMsg.content) {
      this.messages.updateData(this.messages.totalCount() - 1, msg)
    } else {
      this.messages.pushData(msg)
    }
    if (clearInput) {
      this.userInput = ''
    }
    this.listScroller.scrollToIndex(this.messages.totalCount() - 1)
    this.isResponding = false
  }

  changeMessage(msg:ChatMessage){
    console.info("test:msg=%s,c=%s",msg.content,this.messages.getString(this.messages.totalCount()-1));
    this.messages.updateData(this.messages.totalCount()-1 , msg)
    console.info("test:msg=%s",msg.content);
    this.listScroller.scrollToIndex(this.messages.totalCount() - 1)
    this.userInput = ''
    this.isResponding = false
  }

  aboutToAppear() {
    if (this.controller) {
      this.controller.chatCtrl = this
    }
    this.submitUserInput.bind(this)
    this.postMessage.bind(this)
    this.setTyping.bind(this)
    this.changeMessage.bind(this)
    if (this.welcomeMessage) {
      this.setDefaultMessage(ChatRole.Assistant, this.welcomeMessage)
    }
  }

  build() {
    Column({ space: 0 }) {
        RelativeContainer() {
          Text(this.title)
            .fontWeight(FontWeight.Bold)
            .fontSize(20)
            .id('titleBar')
            .alignRules({
              center: { anchor: '__container__', align: VerticalAlign.Center },
              middle: { anchor: '__container__', align: HorizontalAlign.Center }
            })

          Button() {
            Image(this.clearChatIcon).width(30).height(30)
          }
          .alignRules({
            center: { anchor: '__container__', align: VerticalAlign.Center },
            right: { anchor: '__container__', align: HorizontalAlign.End }
          })
          .margin({ right: 10 })
          .id('clearBtn')
          .onClick((evt) => {
            if (this.onClear) {
              this.onClear(evt);
            }
            this.setDefaultMessage(ChatRole.Assistant, this.welcomeMessage)
          })
          .backgroundColor('#0fff')

          // if (this.needBackButton) {
          //   Button() {
          //     Image(this.backIcon).width(30).height(30)
          //   }
          //   .alignRules({
          //     center: { anchor: '__container__', align: VerticalAlign.Center },
          //     left: { anchor: '__container__', align: HorizontalAlign.Start }
          //   })
          //   .margin({ left: 10 })
          //   .id('backBtn')
          //   .onClick(() => {
          //     router.back()
          //   })
          //   .backgroundColor('#0fff')
          // }
        }
        .height(60)
        .border({ width: { top: 0, bottom: 1 }, color: Color.White })
        .width('100%')

      List({ scroller: this.listScroller }) {
        LazyForEach(this.messages, (item: ChatMessage) => {
          ListItem() {
            if (item.role == ChatRole.User.toString()) {
              Row() {
                Blank()
                  .width(55)
                Column() {
                  if (item.picurl) {
                    Image(item.picurl)

                  } else {
                    if (this.useMarkdown) {
                      lvMarkdownIn({
                        text: item.content,
                        textStyle: {
                          textSize: this.messageFontSize,
                          textColor: this.userMessageTextColor as string,
                          textMarkBackground: '#00000000'
                        }
                      })

                    } else {
                      Text(item.content)
                        .fontSize(this.messageFontSize)
                        .textAlign(TextAlign.Start)
                        .width('100%')
                        .fontColor(this.userMessageTextColor)
                    }
                  }
                }
                .backgroundColor($r('app.color.userMessageBackgroundColor'))
                .layoutWeight(1)
                //.height(80)
                .padding(15)
                .margin({ right: 15 })
                .borderRadius(5)
                // .border({
                //   width : 1,
                //   color : Color.Gray,
                //   radius : 5,
                //   style : BorderStyle.Solid
                // })
                // .border({
                //   width: {
                //     top: 1,
                //     right: 1,
                //     bottom: 3,
                //     left: 1
                //   },
                //   color: {
                //     top: '#f0f0f0',
                //     right: '#f0f0f0',
                //     bottom: '#a0a0a0',
                //     left: '#f0f0f0'
                //   }
                // })
                .shadow({
                  radius: 5,        // 阴影模糊半径
                  color: Color.Gray, // 阴影颜色
                  offsetX: 0,        // X轴偏移
                  offsetY: 4         // Y轴偏移（向下偏移模拟光照）
                })

                Button() {
                  Image(this.userAvatar)
                    .width(40)
                    .height(40)
                }
                .type(ButtonType.Circle)
                .backgroundColor(Color.Transparent)
                .width(40)
                .height(40)
              }
              .justifyContent(FlexAlign.End)
              .padding(10)
              .alignItems(VerticalAlign.Top)
              .width('100%')
            } else {
              Row() {
                Button() {
                  Image(this.botAvatar)
                    .width(40)
                    .height(40)
                }
                .type(ButtonType.Circle)
                .width(40)
                .height(40)
                .backgroundColor(Color.Transparent)

                Column() {
                  if (this.isResponding && !item.content?.length) {
                    LoadingProgress()
                      .color(this.botMessageTextColor)
                      .width(40)
                      .height(40)
                  } else {
                    if (item.picurl) {
                      Image(item.picurl)

                    } else {
                      if (this.useMarkdown) {
                        lvMarkdownIn({
                          text: item.content,
                          textStyle: {
                            textSize: this.messageFontSize,
                            textColor: this.userMessageTextColor as string,
                            textMarkBackground: '#00000000'
                          }
                        })
                      } else {
                        Text(item.content)
                          .fontSize(this.messageFontSize)
                          .textAlign(TextAlign.Start)
                          .width('100%')
                          .fontColor(this.botMessageTextColor)
                      }
                    }
                  }
                }
                .alignItems(HorizontalAlign.Start)
                .backgroundColor($r('app.color.botMessageBackgroundColor'))
                .layoutWeight(1)
                //.height(80)
                .padding(15)
                .margin({ left: 15 })
                .borderRadius(5)
                .shadow({
                  radius: 5,        // 阴影模糊半径
                  color: Color.Gray, // 阴影颜色
                  offsetX: 0,        // X轴偏移
                  offsetY: 4         // Y轴偏移（向下偏移模拟光照）
                })

                Blank()
                  .width(55)
              }
              .alignItems(VerticalAlign.Top)
              .padding(10)
              .width('100%')
            }

          }

        })

      }
      .layoutWeight(1)
      .cachedCount(3)
      .scrollBar(BarState.Auto)

      // if (this.needInputControl) {
      //   Row() {
      //     this.InputControl()
      //
      //   }
      //   .margin({ bottom: 10 })
      //   //.borderWidth({ top: 1 })
      //   .borderColor('#999')
      //   .padding(5)
      //   .width('100%')
      //   .height(150)
      // }

      Column(){
        Row() {
          Button() {
            Image($r('app.media.model_choose')).width(25).height(25)
          }
          .backgroundColor(Color.White)
          .aspectRatio(1)        // 宽高比1:1 → 高度=宽度
          .width('30')
          .margin(5)
          .onClick(() => {
            this.getUIContext().showTextPickerDialog({
              range: this.models,
              selected: this.selected_model_index,
              value: this.selected_model_name,
              defaultPickerItemHeight: 40,
              alignment: DialogAlignment.Center,
              offset: { dx: 0, dy: 0 },
              canLoop: false,
              onAccept: (value: TextPickerResult) => {
                // 设置select为按下确定按钮时候的选中项index，这样当弹窗再次弹出时显示选中的是上一次确定的选项
                this.selected_model_index = value.index;
                console.log(this.selected_model_index + '');
                // 点击确定后，被选到的文本数据展示到页面
                this.selected_model_name = value.value as string;
                this.hasSelect = true;
                console.info("TextPickerDialog:onAccept()" + JSON.stringify(value));
              },
              onCancel: () => {
                this.hasSelect = false;
                console.info("TextPickerDialog:onCancel()");
              },
              onChange: (value: TextPickerResult) => {
                console.info("TextPickerDialog:onChange()" + JSON.stringify(value));
              },
              onScrollStop: (value: TextPickerResult) => {
                console.info("TextPickerDialog:onScrollStop()" + JSON.stringify(value));
              },
              onDidAppear: () => {
                console.info("TextPickerDialog:onDidAppear()");
              },
              onDidDisappear: () => {
                console.info("TextPickerDialog:onDidDisappear()");
              },
              onWillAppear: () => {
                console.info("TextPickerDialog:onWillAppear()");
              },
              onWillDisappear: () => {
                console.info("TextPickerDialog:onWillDisappear()");
              }
            });
          })
          // .onClick(() => {
          //   ActionSheet.show({
          //     title: '您的模型',
          //     message: '请选择所需模型',
          //     sheets: [
          //       { title: 'Qwen3', action: () => {
          //         console.log('Get Alert Dialog handled')
          //       } },
          //       { title: 'Stable Diffusion', action: () => {
          //         console.log('Get Alert Dialog handled')
          //       } }
          //     ],
          //     confirm: {
          //       value: '取消',
          //       action: () => console.log('已取消')
          //     }
          //
          //   })
          // })
          Button(){
            Image($r('app.media.add_more')).width(25).height(25)
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
        .layoutWeight(1)
        //.margin(5)

        Row(){
          TextArea({
            placeholder: CONST.InputPropmt,
            controller: this.Textcontroller
          })
            .enterKeyType(EnterKeyType.Send)
            .fontColor(Color.Black)
            .backgroundColor($r('app.color.input_background'))
            .width('75%')
            .flexGrow(1)
            .onChange((val) => {
              this.userInput = val
            })
            .onSubmit(() => {
              if(this.hasSelect){
                this.submitUserInput(this.userInput)
                //this.sendMessage(this.chatController.chatCtrl,this.userInput)
                this.Textcontroller.stopEditing()
                this.Textcontroller.deleteText()
              }else {
                this.getUIContext().showAlertDialog({
                  title: '提示',
                  message: '请先点击输入栏上方最左侧按钮，选择对话模型',
                  autoCancel: true,
                  alignment: DialogAlignment.Center,
                  offset: { dx: 0, dy: 0 },
                  gridCount: 3,
                  width: 300,
                  height: 200,
                  //cornerRadius: 20,
                  // borderWidth: 1,
                  // borderStyle: BorderStyle.Solid, //使用borderStyle属性，需要和borderWidth属性一起使用
                  // borderColor: Color.Black, //使用borderColor属性，需要和borderWidth属性一起使用
                  backgroundColor: Color.White,
                  // shadow: ({
                  //   radius: 20,
                  //   color: Color.Grey,
                  //   offsetX: 50,
                  //   offsetY: 0
                  // }),
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
              }
            })
            // .cancelButton({
            //   style: CancelButtonStyle.INPUT,
            //   icon: {
            //     size: 20,
            //     src: $r('app.media.InputDelete'),
            //     color: Color.Gray
            //   }
            // })
            .caretColor(Color.Black)
            .fontSize(12)
            .wordBreak(WordBreak.BREAK_ALL)
            .height('100%')
            .margin({ top: 0, right: 5, bottom: 0, left: 5 })
          //.padding({ top: 5, right: 5, bottom: 0, left: 5 })

          Button(){
            Image($r('app.media.SendButton')).width(30).height(30)
          }
          .height('100%')
          .type(ButtonType.Circle)
          .backgroundColor($r('app.color.sendbutton_background'))
          .onClick(() => {
            if(this.hasSelect){
              this.submitUserInput(this.userInput)
              //this.sendMessage(this.chatController.chatCtrl,this.userInput)
              this.Textcontroller.stopEditing()
              this.Textcontroller.deleteText()
            }else {
              this.getUIContext().showAlertDialog({
                title: '提示',
                message: '请先点击输入栏上方最左侧按钮，选择对话模型',
                autoCancel: true,
                alignment: DialogAlignment.Center,
                offset: { dx: 0, dy: 0 },
                gridCount: 3,
                width: 300,
                height: 200,
                //cornerRadius: 20,
                // borderWidth: 1,
                // borderStyle: BorderStyle.Solid, //使用borderStyle属性，需要和borderWidth属性一起使用
                // borderColor: Color.Black, //使用borderColor属性，需要和borderWidth属性一起使用
                backgroundColor: Color.White,
                // shadow: ({
                //   radius: 20,
                //   color: Color.Grey,
                //   offsetX: 50,
                //   offsetY: 0
                // }),
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
            }
          })
        }
        .layoutWeight(1)
        .width('100%')

      }
        .margin({ bottom: 10 })
        //.borderWidth({ top: 1 })
        //.borderColor('#999')
        .padding(5)
        .width('100%')
        .height(150)
    }
    .backgroundColor(Color.White)
    .height('100%')
    .width('100%')
  }

  // aboutToDisappear() {
  //   if (this.controller) {
  //     this.controller = undefined
  //   }
  //
  // }
}

export { Chat, IChatComponent, ChatController }