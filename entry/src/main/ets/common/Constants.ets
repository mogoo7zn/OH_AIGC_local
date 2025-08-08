import { relationalStore } from '@kit.ArkData';
import { ChatHistoryTable } from '../viewmodel/ConstantsInterface'

export class Constants {
  static InputPropmt = 'Chat with your local AI now ...'
  static model_uri : string = 'file://docs/storage/Users/currentUser/Download/qwen2.5-3b-instruct-q8_0.gguf'

  static readonly DATABASE_NAME: string = 'ChatHistory.db';
  static readonly DATABASE_VERSION: number = 1;

  static getTokenizer(): relationalStore.Tokenizer | undefined {
    let tokenType = relationalStore.Tokenizer.ICU_TOKENIZER;
    let tokenTypeSupported = relationalStore.isTokenizerSupported(tokenType);
    if (!tokenTypeSupported) {
      console.error(`ICU_TOKENIZER 在该平台不支持`);
      return undefined;
    }
    return tokenType;
  }

  /**
   * Rdb database config.
   */
  static readonly STORE_CONFIG: relationalStore.StoreConfig = {
    name: Constants.DATABASE_NAME,
    securityLevel: relationalStore.SecurityLevel.S1,
    encrypt: false,
    isReadOnly: false,
    tokenizer: Constants.getTokenizer()
  };
  /**
   * ChatHistory table config.
   */
  static readonly CHAT_HISTORY_TABLE: ChatHistoryTable = {
    tableName: 'ChatHistoryTable',
    sqlCreate: 'CREATE TABLE IF NOT EXISTS CHAT_HISTORY_TABLE (ID INTEGER PRIMARY KEY AUTOINCREMENT, ' +
      'SESSION_ID INTEGER, CONVERSATION_ID INTEGER, USER_INPUT TEXT, BOT_OUTPUT TEXT, MODEL_NAME TEXT)',
    columns: ['ID', 'SESSION_ID', 'CONVERSATION_ID', 'USER_INPUT', 'BOT_OUTPUT', 'MODEL_NAME']
  };
  /**
   * Log tag.
   */
  static readonly RDB_TAG = '[Debug.Rdb]';
  static readonly TABLE_TAG = '[Debug.ChatHistoryTable]';
  static readonly INDEX_TAG = '[Debug.Index]';
}

export default Constants;

