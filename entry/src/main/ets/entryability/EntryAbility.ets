import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { relationalStore } from '@kit.ArkData';

import { preferences } from '@kit.ArkData';
import { BusinessError } from '@kit.BasicServicesKit';
import { fileShare } from '@kit.CoreFileKit';
import { Constants as CONST} from '../common/Constants'
import ChatHistoryTable from '../common/database/tables/ChatHistoryTable'


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

const DOMAIN = 0x0000;
let dataPreferences: preferences.Preferences | null = null;

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onCreate');
  }

  onDestroy(): void {
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    // Main window is created, set main page for this ability
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onWindowStageCreate');

    preferences.getPreferences(this.context, 'myStore', (err: BusinessError, val: preferences.Preferences) => {
      if (err) {
        console.error("Failed to get preferences. code =" + err.code + ", message =" + err.message);
        return;
      }
      dataPreferences = val;
      console.info("Succeeded in getting preferences.");
    })

    // 激活模型文件的使用权限
    activatePermissionExample(CONST.model_uri);

    // let tokenType = relationalStore.Tokenizer.ICU_TOKENIZER;
    // let tokenTypeSupported = relationalStore.isTokenizerSupported(tokenType);
    // if (!tokenTypeSupported) {
    //   console.error(`ICU_TOKENIZER 在该平台不支持`);
    // }
    // const STORE_CONFIG: relationalStore.StoreConfig = {
    //   name: 'ChatHistory.db',
    //   securityLevel: relationalStore.SecurityLevel.S1,
    //   encrypt: false,
    //   customDir: 'data',
    //   isReadOnly: false,
    //   tokenizer: tokenType
    // };

    // const SQL_CREATE_TABLE =
    //   'CREATE TABLE IF NOT EXISTS CHAT_SESSIONS (ID INTEGER PRIMARY KEY AUTOINCREMENT, SESSION_ID INTEGER,' +
    //     ' CONVERSATION_ID INTEGER, USER_INPUT TEXT, BOT_OUTPUT TEXT, MODEL_NAME TEXT)';
    // relationalStore.getRdbStore(this.context, STORE_CONFIG, async (err, store) => {
    //   if (err) {
    //     console.error(`Failed to get RdbStore. Code:${err.code}, message:${err.message}`);
    //     return;
    //   }
    //   console.info('Succeeded in getting RdbStore.');
    //   let storeVersion = store.version;
    //   if (storeVersion === 0) {
    //     try {
    //       await store.execute(SQL_CREATE_TABLE);
    //       storeVersion = 1;
    //     } catch (e) {
    //       const err = e as BusinessError;
    //       console.error(`Failed to execute sql. Code:${err.code}, message:${err.message}`);
    //     }
    //   }
    // });

    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(DOMAIN, 'testTag', 'Succeeded in loading the content.');
    });
  }

  onWindowStageDestroy(): void {
    // Main window is destroyed, release UI related resources
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    // Ability has brought to foreground
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    // Ability has back to background
    hilog.info(DOMAIN, 'testTag', '%{public}s', 'Ability onBackground');
  }
}