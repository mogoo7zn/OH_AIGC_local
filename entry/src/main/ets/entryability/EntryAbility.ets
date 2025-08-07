import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { relationalStore } from '@kit.ArkData';
import { BusinessError } from '@kit.BasicServicesKit';

const DOMAIN = 0x0000;

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

    let tokenType = relationalStore.Tokenizer.ICU_TOKENIZER;
    let tokenTypeSupported = relationalStore.isTokenizerSupported(tokenType);
    if (!tokenTypeSupported) {
      console.error(`ICU_TOKENIZER 在该平台不支持`);
    }
    const STORE_CONFIG: relationalStore.StoreConfig = {
      name: 'ChatHistory.db',
      securityLevel: relationalStore.SecurityLevel.S1,
      encrypt: false,
      customDir: 'data',
      isReadOnly: false,
      tokenizer: tokenType
    };

    const SQL_CREATE_TABLE =
      'CREATE TABLE IF NOT EXISTS CHAT_SESSIONS (ID INTEGER PRIMARY KEY AUTOINCREMENT, SESSION_ID INTEGER,' +
        ' CONVERSATION_ID INTEGER, USER_INPUT TEXT, BOT_OUTPUT TEXT, MODEL_NAME TEXT)';
    relationalStore.getRdbStore(this.context, STORE_CONFIG, async (err, store) => {
      if (err) {
        console.error(`Failed to get RdbStore. Code:${err.code}, message:${err.message}`);
        return;
      }
      console.info('Succeeded in getting RdbStore.');
      let storeVersion = store.version;
      if (storeVersion === 0) {
        try {
          await store.execute(SQL_CREATE_TABLE);
          storeVersion = 1;
        } catch (e) {
          const err = e as BusinessError;
          console.error(`Failed to execute sql. Code:${err.code}, message:${err.message}`);
        }
      }
    });

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