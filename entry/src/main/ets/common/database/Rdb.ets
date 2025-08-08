import { relationalStore } from "@kit.ArkData";
import Logger from '../utils/Logger';
import Constants from '../Constants';


export default class Rdb {
  private rdbStore: relationalStore.RdbStore | null = null;
  private tableName: string;
  private sqlCreateTable: string;
  private columns: Array<string>;

  constructor(tableName: string, sqlCreateTable: string, columns: Array<string>) {
    this.tableName = tableName;
    this.sqlCreateTable = sqlCreateTable;
    this.columns = columns;
  }

  /**
   *
   * @param callback
   */
  getRdbStore(callback: Function = () => {})
  {
    if (!callback || typeof callback === 'undefined' || callback === undefined) {
      Logger.info(Constants.RDB_TAG, 'getRdbStore() 没有回调!');
      return;
    }
    if (this.rdbStore !== null) {
      Logger.info(Constants.RDB_TAG, "数据库已存在");
      callback();
      return;
    }
    let context: Context = getContext(this) as Context;
    relationalStore.getRdbStore(context, Constants.STORE_CONFIG, (err, rdb) => {
      if (err) {
        Logger.error(Constants.RDB_TAG, `gerRdbStore() 失败, 错误原因: ${err}`);
        return;
      }
      this.rdbStore = rdb;
      this.rdbStore.executeSql(this.sqlCreateTable);
      Logger.info(Constants.RDB_TAG, 'getRdbStore() 成功！');
      callback();
    })
  }

  /**
   * 插入数据
   * @param data
   * @param callback
   */
  insertData(data: relationalStore.ValuesBucket, callback: Function = () => {}) {
    if (!callback || typeof callback === 'undefined' || callback === undefined) {
      Logger.info(Constants.RDB_TAG, 'insertData() 没有回调!!');
      return;
    }
    let resFlag: boolean = false; //
    const valueBuckets: relationalStore.ValuesBucket = data;
    if (this.rdbStore) {
      this.rdbStore.insert(this.tableName, valueBuckets, (err, ret) => {
        if (err) {
          Logger.error(Constants.RDB_TAG, `insertData() 失败, 错误信息: ${err}`);
          callback(resFlag);
          return;
        }
        Logger.info(Constants.RDB_TAG, `insertData() 成功: ${ret}`);
        callback(ret);
      })
    }
  }

  /**
   *
   * @param predicates
   * @param callback
   */
  deleteData(predicates: relationalStore.RdbPredicates, callback: Function = () => {
  }) {
    if (!callback || typeof callback === 'undefined' || callback === undefined) {
      Logger.info(Constants.RDB_TAG, 'deleteData() 没有回调!');
      return;
    }
    let resFlag: boolean = false;
    if (this.rdbStore) {
      this.rdbStore.delete(predicates, (err, ret) => {
        if (err) {
          Logger.error(Constants.RDB_TAG, `deleteData() 失败, 错误信息: ${err}`);
          callback(resFlag);
          return;
        }
        Logger.info(Constants.RDB_TAG, `deleteData() 成功: ${ret}`);
        callback(!resFlag);
      });
    }
  }

  /**
   *
   * @param predicates
   * @param data
   * @param callback
   */
  updateData(predicates: relationalStore.RdbPredicates, data: relationalStore.ValuesBucket, callback: Function = () => {
  }) {
    if (!callback || typeof callback === 'undefined' || callback === undefined) {
      Logger.info(Constants.RDB_TAG, 'updateDate() has no callback!');
      return;
    }
    let resFlag: boolean = false;
    const valueBucket: relationalStore.ValuesBucket = data;
    if (this.rdbStore) {
      this.rdbStore.update(valueBucket, predicates, (err, ret) => {
        if (err) {
          Logger.error(Constants.RDB_TAG, `updateData() failed, err: ${err}`);
          callback(resFlag);
          return;
        }
        Logger.info(Constants.RDB_TAG, `updateData() finished: ${ret}`);
        callback(!resFlag);
      });
    }
  }

  /**
   * 查询
   * @param predicates
   * @param callback
   */
  query(predicates: relationalStore.RdbPredicates, callback: Function = () => {})
  {
    if (!callback || typeof callback === 'undefined' || callback === undefined) {
      Logger.info(Constants.RDB_TAG, 'query() 没有回调!');
      return;
    }
    if (this.rdbStore) {
      this.rdbStore.query(predicates, this.columns, (err, resultSet) => {
        if (err) {
          Logger.error(Constants.RDB_TAG, `query() 失败, 错误信息:  ${err}`);
          return;
        }
        Logger.info(Constants.RDB_TAG, 'query() 成功');
        callback(resultSet);
        resultSet.close();
      });
    }
  }


}