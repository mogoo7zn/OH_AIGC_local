import { relationalStore } from '@kit.ArkData';
import { ChatHistoryTable, SessionTable, ModelPathTable } from '../viewmodel/ConstantsInterface'

export class Constants {
  static InputPropmt = 'Chat with your local AI now ...'

  static readonly DatabaseName: string = 'ChatHistory.db';
  static readonly DatabaseVersion: number = 1;

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
   * Rdb database config
   */
  static readonly StoreConfig: relationalStore.StoreConfig = {
    name: Constants.DatabaseName,
    securityLevel: relationalStore.SecurityLevel.S1,
    encrypt: false,
    isReadOnly: false,
    tokenizer: Constants.getTokenizer()
  };
  /**
   * ChatHistory table config
   */
  static readonly ChatHistoryTable: ChatHistoryTable = {
    tableName: 'ChatHistoryTable',
    sqlCreate: 'CREATE TABLE IF NOT EXISTS ChatHistoryTable (id INTEGER PRIMARY KEY AUTOINCREMENT, ' +
      'sessionId INTEGER, conversationId INTEGER, userInput TEXT, botOutput TEXT)',
    columns: ['id', 'sessionId', 'conversationId', 'userInput', 'botOutput']
  };

  /**
   * Session table config
   */
  static readonly SessionTable: SessionTable = {
    tableName: 'SessionTable',
    sqlCreate: 'CREATE TABLE IF NOT EXISTS SessionTable (id INTEGER PRIMARY KEY AUTOINCREMENT, ' +
      'sessionName TEXT, prompt TEXT, modelName TEXT)',
    columns: ['id', 'sessionName', 'prompt', 'modelName']
  }

  /**
   * ModelPath table config
   */
  static readonly ModelPathTable: ModelPathTable = {
    tableName: 'ModelPathTable',
    sqlCreate: 'CREATE TABLE IF NOT EXISTS ModelPathTable (id INTEGER PRIMARY KEY AUTOINCREMENT, ' +
      'modelFileUrl TEXT, modelName TEXT)',
    columns: ['id', 'modelFileUrl', 'modelName']
  }

  /**
   * Log tag.
   */
  static readonly RDB_TAG = '[Debug.Rdb]';
  static readonly TABLE_TAG = '[Debug.ChatHistoryTable]';
  static readonly INDEX_TAG = '[Debug.Index]';
}

export default Constants;