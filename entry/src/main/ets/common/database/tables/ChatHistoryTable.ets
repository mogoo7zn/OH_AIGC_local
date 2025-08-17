import { relationalStore } from '@kit.ArkData';
import ChatHistoryData from '../../../viewmodel/ChatHistoryData';
import Constants from '../../Constants';
import Rdb from '../Rdb';


/**
 * 历史对话表
 */
export default class ChatHistoryTable {
  private chatHistoryTable = new Rdb(Constants.ChatHistoryTable.tableName, Constants.ChatHistoryTable.sqlCreate,
    Constants.ChatHistoryTable.columns);

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
    let predicates = new relationalStore.RdbPredicates(Constants.ChatHistoryTable.tableName);
    predicates.equalTo('id', chatHistory.id);
    this.chatHistoryTable.deleteData(predicates, callback);
  }

  updateData(chatHistory:ChatHistoryData, callback:Function)
  {
    const valueBucket: relationalStore.ValuesBucket = generateBucket(chatHistory);
    let predicates = new relationalStore.RdbPredicates(Constants.ChatHistoryTable.tableName);
    predicates.equalTo('id', chatHistory.id);
    this.chatHistoryTable.updateData(predicates, valueBucket, callback);
  }

  query(session_id: number, conversation_id: number, user_input: string, bot_output: string, model_name: string, callback: Function, isAll: boolean = true)
  {
    let predicates = new relationalStore.RdbPredicates(Constants.ChatHistoryTable.tableName);
    if (!isAll) {
      predicates.equalTo('sessionId', session_id);
      predicates.equalTo('conversationId', conversation_id);
      predicates.equalTo('userInput', user_input);
      predicates.equalTo('botOutput', bot_output);
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
            id: 0, sessionId: 0, conversationId: 0, userInput:'', botOutput: ''
          }
          tmp.id = resultSet.getDouble(resultSet.getColumnIndex('id'));
          tmp.sessionId = resultSet.getDouble(resultSet.getColumnIndex('sessionId'));
          tmp.conversationId = resultSet.getDouble(resultSet.getColumnIndex('conversationId'));
          tmp.userInput = resultSet.getString(resultSet.getColumnIndex('userInput'));
          tmp.botOutput = resultSet.getString(resultSet.getColumnIndex('botOutput'));
          result[i] = tmp;
          resultSet.goToNextRow();
        }
        callback(result);
      }
    })
  }

  // queryBySessionId(sessionId: number, callback: Function) {
  //   let predicates = new relationalStore.RdbPredicates(Constants.ChatHistoryTable.tableName);
  //   predicates.equalTo('sessionId', sessionId);
  //   predicates.orderBy('conversationId', relationalStore.OrderDirection.ASC);
  //
  //   this.chatHistoryTable.query(predicates, (resultSet: relationalStore.ResultSet) => {
  //     let count: number = resultSet.rowCount;
  //     if (count === 0 || typeof count === 'string') {
  //       console.log(`${Constants.TABLE_TAG} 查询无结果`);
  //       callback([]);
  //     } else {
  //       resultSet.goToFirstRow();
  //       const result: ChatHistoryData[] = [];
  //       for (let i = 0; i < count; i++) {
  //         let tmp: ChatHistoryData = new ChatHistoryData();
  //         tmp.id = resultSet.getDouble(resultSet.getColumnIndex('id'));
  //         tmp.sessionId = resultSet.getDouble(resultSet.getColumnIndex('sessionId'));
  //         tmp.conversationId = resultSet.getDouble(resultSet.getColumnIndex('conversationId'));
  //         tmp.userInput = resultSet.getString(resultSet.getColumnIndex('userInput'));
  //         tmp.botOutput = resultSet.getString(resultSet.getColumnIndex('botOutput'));
  //         result[i] = tmp;
  //         resultSet.goToNextRow();
  //       }
  //       callback(result);
  //     }
  //   });
  // }
  //
  // deleteBySessionId(sessionId: number, callback: Function) {
  //   let predicates = new relationalStore.RdbPredicates(Constants.ChatHistoryTable.tableName);
  //   predicates.equalTo('sessionId', sessionId);
  //   this.chatHistoryTable.deleteData(predicates, callback);
  // }

  
}

function generateBucket(chatHistory: ChatHistoryData): relationalStore.ValuesBucket {
  let obj: relationalStore.ValuesBucket = {};
  obj.session_id = chatHistory.sessionId;
  obj.conversation_id = chatHistory.conversationId;
  obj.user_input = chatHistory.userInput;
  obj.bot_output = chatHistory.botOutput;
  return obj;
}
