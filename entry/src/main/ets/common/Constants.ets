import { relationalStore } from '@kit.ArkData';
import { ChatHistoryTable, SessionTable, ModelPathTable } from '../viewmodel/ConstantsInterface'

export class Constants {
  static InputPropmt = 'Chat with your local AI now ...'
  static model_uri : string = 'file://docs/storage/Users/currentUser/Download/Qwen3-4B-Instruct-2507-UD-Q6_K_XL.gguf'
  static prompt =
  '你是一个智能助手。\n'
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

  static readonly DatabaseName: string = 'ChatHistory.db';
  static readonly DatabaseVersion: number = 1;

  static getTokenizer(): relationalStore.Tokenizer | undefined {
    let tokenType = relationalStore.Tokenizer.ICU_TOKENIZER;
    // let tokenTypeSupported = relationalStore.isTokenizerSupported(tokenType);
    // if (!tokenTypeSupported) {
    //   console.error(`ICU_TOKENIZER 在该平台不支持`);
    //   return undefined;
    // }
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

