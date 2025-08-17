import { relationalStore } from '@kit.ArkData';
import SessionData from '../../../viewmodel/SessionData';
import Constants from '../../Constants';
import Rdb from '../Rdb';

/**
 * 会话表
 */
export default class SessionTable {
  private sessionTable = new Rdb(Constants.SessionTable.tableName, Constants.SessionTable.sqlCreate,
    Constants.SessionTable.columns);

  constructor(callback: Function = () => {}) {
    this.sessionTable.getRdbStore(callback);
  }

  getRdbStore(callback: Function = () => {}) {
    this.sessionTable.getRdbStore(callback);
  }

  insertData(session: SessionData, callback: Function) {
    const valueBucket: relationalStore.ValuesBucket = generateBucket(session);
    this.sessionTable.insertData(valueBucket, callback);
  }

  deleteData(session: SessionData, callback: Function) {
    let predicates = new relationalStore.RdbPredicates(Constants.SessionTable.tableName);
    predicates.equalTo('id', session.id);
    this.sessionTable.deleteData(predicates, callback);
  }

  updateData(session: SessionData, callback: Function) {
    const valueBucket: relationalStore.ValuesBucket = generateBucket(session);
    let predicates = new relationalStore.RdbPredicates(Constants.SessionTable.tableName);
    predicates.equalTo('id', session.id);
    this.sessionTable.updateData(predicates, valueBucket, callback);
  }

  query(id: number, sessionName: string, callback: Function, isAll: boolean = true) {
    let predicates = new relationalStore.RdbPredicates(Constants.SessionTable.tableName);
    if (!isAll) {
      if (id !== -1) {
        predicates.equalTo('id', id);
      }
      if (sessionName !== '') {
        predicates.equalTo('sessionName', sessionName);
      }
    }
    this.sessionTable.query(predicates, (resultSet: relationalStore.ResultSet) => {
      let count: number = resultSet.rowCount;
      if (count === 0 || typeof count === 'string') {
        console.log(`${Constants.TABLE_TAG}` + '查询无结果');
        callback([]);
      } else {
        resultSet.goToFirstRow();
        const result: SessionData[] = [];
        for (let i = 0; i < count; i++) {
          let tmp: SessionData = new SessionData();
          tmp.id = resultSet.getDouble(resultSet.getColumnIndex('id'));
          tmp.sessionName = resultSet.getString(resultSet.getColumnIndex('sessionName'));
          tmp.prompt = resultSet.getString(resultSet.getColumnIndex('prompt'));
          tmp.modelName = resultSet.getString(resultSet.getColumnIndex('modelName'));
          result[i] = tmp;
          resultSet.goToNextRow();
        }
        callback(result);
      }
    })
  }
}

function generateBucket(session: SessionData): relationalStore.ValuesBucket {
  let obj: relationalStore.ValuesBucket = {};
  obj.sessionName = session.sessionName;
  obj.prompt = session.prompt;
  obj.modelName = session.modelName;
  return obj;
}