import { Constants as CONST} from '../common/Constants'
import { hilog } from '@kit.PerformanceAnalysisKit';
import { picker } from '@kit.CoreFileKit';
import { fileShare } from '@kit.CoreFileKit';
import { UIAbility } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { preferences } from '@kit.ArkData';
import router from '@ohos.router';

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


if (!canIUse('SystemCapability.FileManagement.AppFileService.FolderAuthorization')) {
  console.error('this api is not supported on this device');
}

export async function Picker_persistPermission() : Promise<string> {
  let uri : string = ''
  try {
    let DocumentSelectOptions = new picker.DocumentSelectOptions();
    DocumentSelectOptions.maxSelectNumber = 1;
    let documentPicker = new picker.DocumentViewPicker();
    let uris = await documentPicker.select(DocumentSelectOptions);

    // for(let i=0;i<uris.length;i++){
    //   console.info("has persisted uris");
    //   console.info("persisted uris:" + uris[i]);
    // }

    let policyInfo: fileShare.PolicyInfo = {
      uri: uris[0],
      operationMode: fileShare.OperationMode.READ_MODE,
    };
    let policies: Array<fileShare.PolicyInfo> = [policyInfo];
    await fileShare.persistPermission(policies).then(() => {
      console.info("persistPermission successfully");
      console.info("persisted uris:" + uris[0]);
      uri = uris[0];
    }).catch((err: BusinessError<Array<fileShare.PolicyErrorResult>>) => {
      console.error("persistPermission failed with error message: " + err.message + ", error code: " + err.code);
      if (err.code == 13900001 && err.data) {
        for (let i = 0; i < err.data.length; i++) {
          console.error("error code : " + JSON.stringify(err.data[i].code));
          console.error("error uri : " + JSON.stringify(err.data[i].uri));
          console.error("error reason : " + JSON.stringify(err.data[i].message));
        }
      }
    });
  } catch (error) {
    let err: BusinessError = error as BusinessError;
    console.error(`persistPermission failed with err, Error code: ${err.code}, message: ${err.message}`);
  }
  return uri;
}

async function activatePermissionExample(uris: string) : Promise<boolean> {
  try {
    //let uri = "file://docs/storage/Users/username/tmp.txt";
    let uri = uris;
    console.info("trying activate uris:" + uris);
    let policyInfo: fileShare.PolicyInfo = {
      uri: uri,
      operationMode: fileShare.OperationMode.READ_MODE,
    };
    let policies: Array<fileShare.PolicyInfo> = [policyInfo];
    fileShare.activatePermission(policies).then(() => {
      console.info("activated uris:" + uri);
      console.info("activatePermission successfully");
    }).catch((err: BusinessError<Array<fileShare.PolicyErrorResult>>) => {
      console.error("activatePermission failed with error message: " + err.message + ", error code: " + err.code);
      if (err.code == 13900001 && err.data) {
        for (let i = 0; i < err.data.length; i++) {
          console.error("error code : " + JSON.stringify(err.data[i].code));
          console.error("error uri : " + JSON.stringify(err.data[i].uri));
          console.error("error reason : " + JSON.stringify(err.data[i].message));
          if (err.data[i].code == fileShare.PolicyErrorCode.PERMISSION_NOT_PERSISTED) {
            //可以选择进行持久化后再激活。
          }
        }
      }
    });
  } catch (error) {
    let err: BusinessError = error as BusinessError;
    console.error(`activatePermission failed with err, Error code: ${err.code}, message: ${err.message}`);
  }
  return true;
}


async function revokePermissionExample() {
  try {
    let uri = "file://docs/storage/Users/username/tmp.txt";
    let policyInfo: fileShare.PolicyInfo = {
      uri: uri,
      operationMode: fileShare.OperationMode.READ_MODE,
    };
    let policies: Array<fileShare.PolicyInfo> = [policyInfo];
    fileShare.revokePermission(policies).then(() => {
      console.info("revokePermission successfully");
    }).catch((err: BusinessError<Array<fileShare.PolicyErrorResult>>) => {
      console.error("revokePermission failed with error message: " + err.message + ", error code: " + err.code);
      if (err.code == 13900001 && err.data) {
        for (let i = 0; i < err.data.length; i++) {
          console.error("error code : " + JSON.stringify(err.data[i].code));
          console.error("error uri : " + JSON.stringify(err.data[i].uri));
          console.error("error reason : " + JSON.stringify(err.data[i].message));
        }
      }
    });
  } catch (error) {
    let err: BusinessError = error as BusinessError;
    console.error(`revokePermission failed with err, Error code: ${err.code}, message: ${err.message}`);
  }
}

@Entry
@Component
struct setting {
  //@Provide('modeluri') model_uri : string = '';

  build() {
    RelativeContainer(){
      Button(){
        Image($r('app.media.BackButton')).width(35).height(35)
      }
      .alignRules({
        center: { anchor: '__container__', align: VerticalAlign.Top },
        left: { anchor: '__container__', align: HorizontalAlign.Start },
      })
      .padding(5)
      .offset({
        y: 25,
        x: 5,
      })
      .id('back button')
      .backgroundColor(Color.White)
      .onClick(()=>{
        this.getUIContext().getRouter().back()
        //   .pushUrl({
        //   url: 'pages/Index' // 目标url
        // }, router.RouterMode.Single, (err) => {
        //   if (err) {
        //     console.error(`Invoke pushUrl failed, code is ${err.code}, message is ${err.message}`);
        //     return;
        //   }
        //   console.info('Invoke pushUrl succeeded.');
        // });
      })

      Column(){
        Text('添加模型')
          .fontSize(20)
          .border({
            width : {
              top: 1,
              bottom: 1
            },
            style : {
              top: BorderStyle.Solid,
              bottom: BorderStyle.Solid
            },
            color : Color.Black
          })
          .padding(10)
          .height(50)
          .width('100%')
          .onClick(()=>{
             Picker_persistPermission()
               .then((uri: string)=>{
                  CONST.model_uri = uri;
               });
            activatePermissionExample(CONST.model_uri); // TODO : 传入参数须为持久化存储
            hilog.info(0, 'testTag', 'model_uri ( in setting ) :' + CONST.model_uri);
          })

        Text('配置prompt')
          .fontSize(20)
          .border({
            width : {
              bottom: 1
            },
            style : {
              top: BorderStyle.Solid,
              bottom: BorderStyle.Solid
            },
            color : Color.Black
          })
          .padding(10)
          .height(50)
          .width('100%')
          .onClick(()=>{
            this.getUIContext().getRouter().pushUrl({
              url: 'pages/PromptSetting' // 目标url
            }, router.RouterMode.Single, (err) => {
              if (err) {
                console.error(`Invoke pushUrl failed, code is ${err.code}, message is ${err.message}`);
                return;
              }
              console.info('Invoke pushUrl succeeded.');
            });
          })
      }
      .width('100%')
      .alignRules({
        top: { anchor: 'back button', align: VerticalAlign.Bottom},
        left: { anchor: '__container__', align: HorizontalAlign.Start },
      })
      .offset({
        y: 25
      })

    }
  }


}