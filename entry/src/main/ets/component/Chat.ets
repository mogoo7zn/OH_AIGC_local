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
import { hilog } from '@kit.PerformanceAnalysisKit';
import { Picker_persistPermission } from '../pages/setting'

import { picker } from '@kit.CoreFileKit';
import { fileShare } from '@kit.CoreFileKit';
import ChatHistoryData from '../viewmodel/ChatHistoryData'
import Rdb from '../common/database/Rdb'
import testNapi from 'libentry.so';
import ModelPathTable from '../common/database/tables/ModelPathTable'
import ModelPath from '../viewmodel/ModelPath'

import {
  PhotoPickerComponent,
  PickerController,
  PickerOptions,
  DataType,
  BaseItemInfo,
  ItemInfo,
  PhotoBrowserInfo,
  ItemType,
  ClickType,
  MaxCountType,
  PhotoBrowserRange,
  PhotoBrowserUIElement,
  ReminderMode,
  ItemsDeletedCallback,
  ExceedMaxSelectedCallback,
  CurrentAlbumDeletedCallback,
  photoAccessHelper,
} from '@kit.MediaLibraryKit';
//import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { fileIo } from '@kit.CoreFileKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { dataSharePredicates } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';


// class MediaDataHandler implements photoAccessHelper.MediaAssetDataHandler<ArrayBuffer> {
//   onDataPrepared(data: ArrayBuffer) {
//     if (data === undefined) {
//       console.error('Error occurred when preparing data');
//       return;
//     }
//     console.info('on image data prepared');
//     // 应用自定义对资源数据的处理逻辑。
//   }
// }
//
// async function example(phAccessHelper: photoAccessHelper.PhotoAccessHelper, context: Context, uri: string) {
//   let predicates: dataSharePredicates.DataSharePredicates = new dataSharePredicates.DataSharePredicates();
//   //let uri = 'file://media/Photo/1/IMG_datetime_0001/displayName.jpg' // 需保证此uri已存在。
//   predicates.equalTo(photoAccessHelper.PhotoKeys.URI, uri.toString());
//   let fetchOptions: photoAccessHelper.FetchOptions = {
//     fetchColumns: [photoAccessHelper.PhotoKeys.TITLE],
//     predicates: predicates
//   };
//
//   try {
//     let fetchResult: photoAccessHelper.FetchResult<photoAccessHelper.PhotoAsset> = await phAccessHelper.getAssets(fetchOptions);
//     let photoAsset: photoAccessHelper.PhotoAsset = await fetchResult.getFirstObject();
//     console.info('getAssets photoAsset.uri : ' + photoAsset.uri);
//     // 获取属性值，以标题为例；对于非默认查询的属性，get前需要在fetchColumns中添加对应列名。
//     console.info('title : ' + photoAsset.get(photoAccessHelper.PhotoKeys.TITLE));
//     // 请求图片资源数据。
//     let requestOptions: photoAccessHelper.RequestOptions = {
//       deliveryMode: photoAccessHelper.DeliveryMode.HIGH_QUALITY_MODE,
//     }
//     await photoAccessHelper.MediaAssetManager.requestImageData(context, photoAsset, requestOptions, new MediaDataHandler());
//     console.info('requestImageData successfully');
//     fetchResult.close();
//   } catch (err) {
//     console.error('getAssets failed with err: ' + err);
//   }
// }




let a = new ChatHistoryData()




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

  postMessageStream(msg: ChatMessage, clearInput?: boolean): void

  deleteMessage() : void

  setTyping(isTyping: boolean): void

  submitUserTextInput(userIputText: string): void

  changeMessage(msg:ChatMessage):void
}

class ChatController {
  chatCtrl?: IChatComponent
}

let modelPathTable = new ModelPathTable();
let modelPath = new ModelPath();

//@Preview
@Component
struct Chat {

  controller = new ChatController()
  Textcontroller: TextAreaController = new TextAreaController()

  private selected_model_index: number | number[] = 0;
  // private models: string[] = ['qwen2.5-3b-instruct-q8_0.gguf', 'qwen2-0_5b-instruct-q4_0.gguf', 'qwen2.5-7b-instruct-q3_k_m.gguf', 'DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf'];
  private models: string[] = []
  private models_uri: string[] = []
  @Prop selected_model_name : string = '';
  @Prop selected_model_uri : string = '';
  @State hasSelect : boolean = false;

  private file_types: string[] = ['图片', '音频'];
  private selected_file_type_index: number | number[] = 0;
  private selected_file_type: string = '';

  // photoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
  // // photoSelectOptions.MIMEType : photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE; // 过滤选择媒体文件类型为IMAGE。
  // // photoSelectOptions.maxSelectNumber = 5; // 选择媒体文件的最大数目。
  //
  // uris: Array<string> = [];
  // photoViewPicker = new photoAccessHelper.PhotoViewPicker();


  // // 组件初始化时设置参数信息。
  // pickerOptions: PickerOptions = new PickerOptions();
  //
  // // 组件初始化完成后，可控制组件部分行为。
  // @State pickerController: PickerController = new PickerController();
  //
  // // 已选择的图片。
  // @State selectUris: Array<string> = new Array<string>();
  //
  // // 目前选择的图片。
  // @State currentUri: string = '';
  //
  // // 是否显示大图。
  // @State isBrowserShow: boolean = false;
  //
  // // 资源被选中回调，返回资源的信息，以及选中方式。
  // private onItemClicked(itemInfo: ItemInfo, clickType: ClickType): boolean {
  //   if (!itemInfo) {
  //     return false;
  //   }
  //   let type: ItemType | undefined = itemInfo.itemType;
  //   let uri: string | undefined = itemInfo.uri;
  //   if (type === ItemType.CAMERA) {
  //     // 点击相机item。
  //     return true; // 返回true则拉起系统相机，若应用需要自行处理则返回false。
  //   } else {
  //     if (clickType === ClickType.SELECTED) {
  //       // 应用做自己的业务处理。
  //       if (uri) {
  //         this.selectUris.push(uri);
  //         this.pickerOptions.preselectedUris = [...this.selectUris];
  //       }
  //       return true; // 返回true则勾选，否则则不响应勾选。
  //     } else {
  //       if (uri) {
  //         this.selectUris = this.selectUris.filter((item: string) => {
  //           return item != uri;
  //         });
  //         this.pickerOptions.preselectedUris = [...this.selectUris];
  //       }
  //     }
  //     return true;
  //   }
  // }
  //
  // // 进入大图的回调。
  // private onEnterPhotoBrowser(photoBrowserInfo: PhotoBrowserInfo): boolean {
  //   this.isBrowserShow = true;
  //   return true;
  // }
  //
  // // 退出大图的回调。
  // private onExitPhotoBrowser(photoBrowserInfo: PhotoBrowserInfo): boolean {
  //   this.isBrowserShow = false;
  //   return true;
  // }
  //
  // // 接收到该回调后，便可通过pickerController相关接口向picker发送数据，在此之前不生效。
  // private onPickerControllerReady(): void {
  // }
  //
  // // 大图左右滑动的回调。
  // private onPhotoBrowserChanged(browserItemInfo: BaseItemInfo): boolean {
  //   this.currentUri = browserItemInfo.uri ?? '';
  //   return true;
  // }
  //
  // // 已勾选图片被删除时的回调。
  // private onSelectedItemsDeleted(baseItemInfos: Array<BaseItemInfo>): void {
  // }
  //
  // // 超过最大选择数量再次点击时的回调。
  // private onExceedMaxSelected(exceedMaxCountType: MaxCountType): void {
  // }
  //
  // // 当前相册被删除时的回调。
  // private onCurrentAlbumDeleted(): void {
  // }

  @State messages: IChatDataSource = new ChatDataSource();
  @State userInput: string = ''
  @State isResponding: boolean = false
  private listScroller: Scroller = new Scroller();
  public onClear?: (event?: ClickEvent) => void;
  public onSendMessage?: (ctl: IChatComponent, message: ChatMessage) => void;
  public botAvatar: Resource = $r('app.media.chatbot')
  public userAvatar: Resource = $r('app.media.me')
  public backIcon: Resource = $r('app.media.BackButton')
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

  submitUserTextInput(userInputText: string) {
    this.userInput = userInputText
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

  submitUserPictureInput(userInputPictureUrl: string) {
    if (!this.isResponding) {
      if (this.onSendMessage) {
        this.isResponding = true
        this.onSendMessage(this, new ChatMessage({
          role: ChatRole.User,
          picurl: userInputPictureUrl
        }))
      }
    }
  }

  submitUserAudioInput(userIputAudioUrl: string) {
    if (!this.isResponding) {
      if (this.onSendMessage) {
        this.isResponding = true
        this.onSendMessage(this, new ChatMessage({
          role: ChatRole.User,
          // picurl: userIputPictureUrl
          // TODO : 增加音频类型，并实现播放组件。（ ? :是否需要持久化权限)
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

  // @Builder
  // PickPhoto(){
  //   PhotoPickerComponent({
  //     pickerOptions: this.pickerOptions,
  //     onItemClicked: (itemInfo: ItemInfo, clickType: ClickType): boolean => this.onItemClicked(itemInfo, clickType),
  //     onEnterPhotoBrowser: (photoBrowserInfo: PhotoBrowserInfo): boolean => this.onEnterPhotoBrowser(photoBrowserInfo),
  //     onExitPhotoBrowser: (photoBrowserInfo: PhotoBrowserInfo): boolean => this.onExitPhotoBrowser(photoBrowserInfo),
  //     onPickerControllerReady: (): void => this.onPickerControllerReady(),
  //     onPhotoBrowserChanged: (browserItemInfo: BaseItemInfo): boolean => this.onPhotoBrowserChanged(browserItemInfo),
  //     onSelectedItemsDeleted: (BaseItemInfo: Array<BaseItemInfo>) => this.onSelectedItemsDeleted(BaseItemInfo),
  //     onExceedMaxSelected: (exceedMaxCountType: MaxCountType) => this.onExceedMaxSelected(exceedMaxCountType),
  //     onCurrentAlbumDeleted: () => this.onCurrentAlbumDeleted(),
  //     pickerController: this.pickerController,
  //   })
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
    this.listScroller.scrollToIndex(this.messages.totalCount() - 1,true,ScrollAlign.END)
    this.isResponding = false
  }

  postMessageStream(msg: ChatMessage, clearInput: boolean = true) {
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
    hilog.info(0, 'testTag', 'Updating message:' + msg.content);
    console.info("Updating message:", msg.content);
    console.info("test:msg=%s,c=%s",msg.content,this.messages.getString(this.messages.totalCount()-1));
    this.messages.updateData(this.messages.totalCount()-1 , msg)

    // this.messages.removeData(this.messages.totalCount()-1)
    // this.messages.pushData(msg)

    // let lastMsg: ChatMessage = this.messages.getData(this.messages.totalCount() - 1)
    // if (!lastMsg.content) {
    //   this.messages.updateData(this.messages.totalCount() - 1, msg)
    // } else {
    //   this.messages.pushData(msg)
    // }
    console.info("test:msg=%s",msg.content);
    this.listScroller.scrollToIndex(this.messages.totalCount() - 1)
    this.userInput = ''
    this.isResponding = false
  }

  aboutToAppear() {
    if (this.controller) {
      this.controller.chatCtrl = this
    }
    this.submitUserTextInput.bind(this)
    this.postMessage.bind(this)
    this.setTyping.bind(this)
    this.changeMessage.bind(this)
    if (this.welcomeMessage) {
      this.setDefaultMessage(ChatRole.Assistant, this.welcomeMessage)
    }

    // // 设置picker宫格页数据类型
    // this.pickerOptions.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_VIDEO_TYPE // 图片和照片都显示。
    // // 最大选择数量。
    // this.pickerOptions.maxSelectNumber = 1;
    // // 超出最大选择数量时。
    // this.pickerOptions.maxSelectedReminderMode = ReminderMode.TOAST;
    // // 是否展示搜索框，默认false。
    // this.pickerOptions.isSearchSupported = true;
    // // 是否支持拍照，默认false。
    // this.pickerOptions.isPhotoTakingSupported = true;
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

          Button({
            type: ButtonType.Circle
          }){
            Image($r('app.media.setting_button')).width(30).height(30)
          }
          .backgroundColor(Color.White)
          .alignRules({
            center: { anchor: '__container__', align: VerticalAlign.Center },
            right: {anchor: 'newchatBtn', align: HorizontalAlign.Start },
          })
          .id('settingBtn')
          .margin({ right: 10 })
          // .offset({ x : -10 })
          .onClick(()=>{
            this.getUIContext().getRouter().pushUrl({
              url: 'pages/setting' // 目标url
            }, router.RouterMode.Single, (err) => {
              if (err) {
                console.error(`Invoke pushUrl failed, code is ${err.code}, message is ${err.message}`);
                return;
              }
              console.info('Invoke pushUrl succeeded.');
            });
          })

          // Button() {
          //   Image(this.clearChatIcon).width(30).height(30)
          // }
          // .alignRules({
          //   center: { anchor: '__container__', align: VerticalAlign.Center },
          //   right: { anchor: '__container__', align: HorizontalAlign.End },
          // })
          // .margin({ right: 10 })
          // .id('clearBtn')
          // .onClick((evt) => {
          //   if (this.onClear) {
          //     this.onClear(evt);
          //   }
          //   this.setDefaultMessage(ChatRole.Assistant, this.welcomeMessage)
          // })
          // .backgroundColor('#0fff')

          Button(){
            Image($r('app.media.NewchatButton')).width(30).height(30)
          }
          .backgroundColor(Color.White)
          .alignRules({
            center: { anchor: '__container__', align: VerticalAlign.Center },
            right: { anchor: '__container__', align: HorizontalAlign.End },
          })
          .margin({ right: 10 })
          .id('newchatBtn')
          //.offset({ x : 40 + 10 })
          .onClick(()=>{
            this.setDefaultMessage(ChatRole.Assistant, this.welcomeMessage);
            testNapi.unload_module();
            setTimeout(() => {
              console.info('delay 1s');
            }, 1000);
            this.getUIContext().getFocusController().requestFocus('TextArea')
            //todo

            // this.getUIContext().getRouter().replaceUrl({
            //   url: 'pages/Index' // 目标url
            // }, router.RouterMode.Standard, (err) => {
            //   if (err) {
            //     console.error(`Invoke replaceUrl failed, code is ${err.code}, message is ${err.message}`);
            //     return;
            //   }
            //   console.info('Invoke replaceUrl succeeded.');
            // });
          })

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
                  .width(80)
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
                .padding({
                  left: 10,
                  right: 5,
                  top:7,
                  bottom:10
                })
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
                // .shadow({
                //   radius: 5,        // 阴影模糊半径
                //   color: Color.Gray, // 阴影颜色
                //   offsetX: 0,        // X轴偏移
                //   offsetY: 4         // Y轴偏移（向下偏移模拟光照）
                // })

                Button() {
                  Image(this.userAvatar)
                    .width(40)
                    .height(40)
                }
                .type(ButtonType.Circle)
                .backgroundColor(Color.Transparent)
                .width(40)
                .height(40)
                .offset({y:5})
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
                .offset({y:5})
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
                //.backgroundColor($r('app.color.botMessageBackgroundColor'))
                .backgroundColor(Color.White)
                .layoutWeight(1)
                //.height(80)
                .padding(10)
                .margin({ left: 15 })
                .borderRadius(5)
                // .shadow({
                //   radius: 5,        // 阴影模糊半径
                //   color: Color.Gray, // 阴影颜色
                //   offsetX: 0,        // X轴偏移
                //   offsetY: 4         // Y轴偏移（向下偏移模拟光照）
                // })

                Blank()
                  .width(15)
              }
              .alignItems(VerticalAlign.Top)
              .padding(10)
              .width('100%')


              /*
               *
               *               Flex({ direction: FlexDirection.Row }) {
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
                }

                Row(){
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
                // .alignItems(VerticalAlign.Top)
                // .padding(10)
                // .width('100%')
              }
              // .height(70)
              // .width('90%')
              // .padding(10)
              // .backgroundColor('#AFEEEE')*/
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
            modelPathTable.query(-1,'',(models: ModelPath[])=>{
              for(let i=0;i<models.length;i++){
                this.models.push(models[i].modelName)
                this.models_uri.push(models[i].modelFileUrl)
                hilog.info(0,"test",'models:%{public}s',this.models_uri[i])
              }
            }, true)
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
                // modelPathTable.query(this.selected_model_index as number,this.selected_model_name,(models: ModelPath[])=>{
                //   this.selected_model_uri = models[this.selected_model_index as number].modelFileUrl;
                // })
                this.selected_model_uri = this.models_uri[this.selected_model_index as number];
                this.hasSelect = true;
                console.info("TextPickerDialog:onAccept()" + JSON.stringify(value));

                const modeluri = this.selected_model_uri
                let file = fileIo.openSync(modeluri, fileIo.OpenMode.READ_ONLY);
                testNapi.load_module(file.path,CONST.prompt);
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
                // todo : 从数据库更新this.momdels
                console.info("TextPickerDialog:onWillAppear()");
              },
              onWillDisappear: () => {
                this.models = [];
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
            this.getUIContext().showTextPickerDialog({
              range: this.file_types,
              selected: this.selected_file_type_index,
              value: this.selected_file_type,
              defaultPickerItemHeight: 40,
              alignment: DialogAlignment.Center,
              offset: { dx: 0, dy: 0 },
              canLoop: false,
              onAccept: (value: TextPickerResult) => {
                // 设置select为按下确定按钮时候的选中项index，这样当弹窗再次弹出时显示选中的是上一次确定的选项
                this.selected_file_type_index = value.index;
                console.log(this.selected_file_type + '');
                // 点击确定后，被选到的文本数据展示到页面
                this.selected_file_type = value.value as string;
                // 若之后要添加更多的文件类型，此处改用switch-case即可
                if(this.selected_file_type_index == 0){

                  try {
                    let PhotoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
                    PhotoSelectOptions.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
                    PhotoSelectOptions.maxSelectNumber = 1;
                    PhotoSelectOptions.isEditSupported = true;
                    PhotoSelectOptions.isSearchSupported = true;

                    let photoPicker = new photoAccessHelper.PhotoViewPicker();

                    photoPicker.select(PhotoSelectOptions).then((PhotoSelectResult: photoAccessHelper.PhotoSelectResult) => {
                      console.info('PhotoViewPicker.select successfully, PhotoSelectResult uri: ' + JSON.stringify(PhotoSelectResult));
                      this.submitUserPictureInput(PhotoSelectResult.photoUris[0])   // 已设置选择数量上限为1

                    }).catch((err: BusinessError) => {
                      console.error(`PhotoViewPicker.select failed with err: ${err.code}, ${err.message}`);
                    });
                  } catch (error) {
                    let err: BusinessError = error as BusinessError;
                    console.error(`PhotoViewPicker failed with err: ${err.code}, ${err.message}`);
                  }

                } else if (this.selected_file_type_index == 1){

                  // TODO : 实现音频上传功能

                  // 请在组件内获取context，确保this.getUIContext().getHostContext()返回结果为UIAbilityContext
                  let context = this.getUIContext().getHostContext() as common.UIAbilityContext;
                  const audioViewPicker = new picker.AudioViewPicker(context);
                  const audioSelectOptions = new picker.AudioSelectOptions();
                  audioViewPicker.select(audioSelectOptions).then((audioSelectResult: Array<string>) => {
                    //文件选择成功后，返回被选中音频的URI结果集。
                    this.submitUserAudioInput(audioSelectResult[0])   // 已设置选择数量上限为1
                    console.info('audioViewPicker.select to file succeed and uri is:' + audioSelectResult[0]);
                  }).catch((err: BusinessError) => {
                    console.error(`Invoke audioViewPicker.select failed, code is ${err.code}, message is ${err.message}`);
                  })

                }
                console.info("TextPickerDialog:onAccept()" + JSON.stringify(value));
              },
              onCancel: () => {
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
          //     title: '选择您要上传的文件类型',
          //     message: '',
          //     sheets: [
          //       { title: '图片', action: () => { } },
          //       { title: '音频', action: () => {} }
          //     ],
          //     confirm: {
          //       value: '取消',
          //       action: () => console.log('已取消')
          //     }
          //   })
          // })
        }
        .alignItems(VerticalAlign.Center) // 垂直居中对齐
        .width('100%')
        .height(50)
        //.margin(5)

        Row(){
          TextArea({
            placeholder: CONST.InputPropmt,
            controller: this.Textcontroller
          })
            .id('TextArea')
            .enterKeyType(EnterKeyType.Send)
            .fontColor(Color.Black)
            .backgroundColor($r('app.color.input_background'))
            .width('75%')
            .flexGrow(1)
            .caretColor(Color.Black)
            .fontSize(14)
            .wordBreak(WordBreak.BREAK_ALL)
            .height('100%')
            .margin({ top: 0, right: 5, bottom: 0, left: 5 })
            //.padding({ top: 5, right: 5, bottom: 0, left: 5 })
            .onChange((val) => {
              this.userInput = val
            })
            .onSubmit(() => {
              if(this.hasSelect){
                this.submitUserTextInput(this.userInput)
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


          Button(){
            Image($r('app.media.SendButton')).width(30).height(30)
          }
          .height('100%')
          .type(ButtonType.Circle)
          .backgroundColor($r('app.color.sendbutton_background'))
          .onClick(() => {
            if(this.hasSelect){
              this.submitUserTextInput(this.userInput)
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
        .height(75)
        .width('100%')

      }
        .margin({ bottom: 10 })
        //.borderWidth({ top: 1 })
        //.borderColor('#999')
        .padding(5)
        .width('100%')
        .height(120)
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