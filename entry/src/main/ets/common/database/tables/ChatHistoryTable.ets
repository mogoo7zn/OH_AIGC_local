import { relationalStore } from '@kit.ArkData';
import ChatHistoryData from '../../../viewmodel/ChatHistoryData';
import Constants from '../../Constants';
import Rdb from '../Rdb';


/**
 * 历史对话表
 */
export default class ChatHistoryTable {
  private chatHistoryTable = new Rdb(Constants.CHAT_HISTORY_TABLE.tableName, Constants.CHAT_HISTORY_TABLE.sqlCreate,
    Constants.CHAT_HISTORY_TABLE.columns);

  constructor(callback: Function = () => {})
  {
    this.chatHistoryTable.getRdbStore(callback);
  }

  getRdbStore(callback: Function = () => {})
  {
    this.chatHistoryTable.getRdbStore(callback);
  }

  insertData(chatHistory: ChatHistoryData, callback: Function)
  {
    const valueBucket: relationalStore.ValuesBucket = generateBucket(chatHistory);
    this.chatHistoryTable.insertData(valueBucket, callback);
  }

  deleteData(chatHistory: ChatHistoryData, callback: Function)
  {
    let predicates = new relationalStore.RdbPredicates(Constants.CHAT_HISTORY_TABLE.tableName);
    predicates.equalTo('id', chatHistory.id);
    this.chatHistoryTable.deleteData(predicates, callback);
  }

  updateData(chatHistory:ChatHistoryData, callback:Function)
  {
    const valueBucket: relationalStore.ValuesBucket = generateBucket(chatHistory);
    let predicates = new relationalStore.RdbPredicates(Constants.CHAT_HISTORY_TABLE.tableName);
    predicates.equalTo('id', chatHistory.id);
    this.chatHistoryTable.updateData(predicates, valueBucket, callback);
  }

  query(session_id: number, conversation_id: number, user_input: string, bot_output: string, model_name: string, callback: Function, isAll: boolean = true)
  {
    let predicates = new relationalStore.RdbPredicates(Constants.CHAT_HISTORY_TABLE.tableName);
    if (!isAll) {
      predicates.equalTo('session_id', session_id);
      predicates.equalTo('conversation_id', conversation_id);
      predicates.equalTo('user_input', user_input);
      predicates.equalTo('bot_output', bot_output);
      predicates.equalTo('model_name', model_name);
    }
    this.chatHistoryTable.query(predicates, (resultSet: relationalStore.ResultSet) => {
      let count: number = resultSet.rowCount;
      if (count === 0 || typeof count === 'string') {
        console.log(`${Constants.TABLE_TAG}` + '查询无结果');
        callback([]);
      } else {
        resultSet.goToFirstRow();
        const result: ChatHistoryData[] = [];
        for (let i = 0; i < count; i++) {
          let tmp: ChatHistoryData = {
            id: 0, session_id: 0, conversation_id: 0, user_input:'', bot_output: '', model_name: ''
          }
          tmp.id = resultSet.getDouble(resultSet.getColumnIndex('id'));
          tmp.session_id = resultSet.getDouble(resultSet.getColumnIndex('session_id'));
          tmp.conversation_id = resultSet.getDouble(resultSet.getColumnIndex('conversation_id'));
          tmp.user_input = resultSet.getString(resultSet.getColumnIndex('user_input'));
          tmp.bot_output = resultSet.getString(resultSet.getColumnIndex('bot_output'));
          tmp.model_name = resultSet.getString(resultSet.getColumnIndex('model_name'));
          result[i] = tmp;
          resultSet.goToNextRow();
        }
        callback(result);
      }
    })
  }
}

function generateBucket(chatHistory: ChatHistoryData): relationalStore.ValuesBucket {
  let obj: relationalStore.ValuesBucket = {};
  obj.session_id = chatHistory.session_id;
  obj.conversation_id = chatHistory.conversation_id;
  obj.user_input = chatHistory.user_input;
  obj.bot_output = chatHistory.bot_output;
  obj.model_name = chatHistory.model_name;
  return obj;
}
