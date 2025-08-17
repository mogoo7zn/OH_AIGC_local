import { relationalStore } from '@kit.ArkData';
import ModelPath from '../../../viewmodel/ModelPath';
import Constants from '../../Constants';
import Rdb from '../Rdb';

/**
 * 模型路径表
 */
export default class ModelPathTable {
  private modelPathTable = new Rdb(Constants.ModelPathTable.tableName, Constants.ModelPathTable.sqlCreate,
    Constants.ModelPathTable.columns);

  constructor(callback: Function = () => {}) {
    this.modelPathTable.getRdbStore(callback);
  }

  getRdbStore(callback: Function = () => {}) {
    this.modelPathTable.getRdbStore(callback);
  }

  insertData(modelPath: ModelPath, callback: Function) {
    const valueBucket: relationalStore.ValuesBucket = generateBucket(modelPath);
    this.modelPathTable.insertData(valueBucket, callback);
  }

  deleteData(modelPath: ModelPath, callback: Function) {
    let predicates = new relationalStore.RdbPredicates(Constants.ModelPathTable.tableName);
    predicates.equalTo('id', modelPath.id);
    this.modelPathTable.deleteData(predicates, callback);
  }

  updateData(modelPath: ModelPath, callback: Function) {
    const valueBucket: relationalStore.ValuesBucket = generateBucket(modelPath);
    let predicates = new relationalStore.RdbPredicates(Constants.ModelPathTable.tableName);
    predicates.equalTo('id', modelPath.id);
    this.modelPathTable.updateData(predicates, valueBucket, callback);
  }

  query(id: number, modelName: string, callback: Function, isAll: boolean = true) {
    let predicates = new relationalStore.RdbPredicates(Constants.ModelPathTable.tableName);
    if (!isAll) {
      if (id !== -1) {
        predicates.equalTo('id', id);
      }
      if (modelName !== '') {
        predicates.equalTo('modelName', modelName);
      }
    }
    this.modelPathTable.query(predicates, (resultSet: relationalStore.ResultSet) => {
      let count: number = resultSet.rowCount;
      if (count === 0 || typeof count === 'string') {
        console.log(`${Constants.TABLE_TAG}` + '查询无结果');
        callback([]);
      } else {
        resultSet.goToFirstRow();
        const result: ModelPath[] = [];
        for (let i = 0; i < count; i++) {
          let tmp = new ModelPath();
          tmp.id = resultSet.getDouble(resultSet.getColumnIndex('id'));
          tmp.modelFileUrl = resultSet.getString(resultSet.getColumnIndex('modelFileUrl'));
          tmp.modelName = resultSet.getString(resultSet.getColumnIndex('modelName'));
          result[i] = tmp;
          resultSet.goToNextRow();
        }
        callback(result);
      }
    })
  }
}

function generateBucket(modelPath: ModelPath): relationalStore.ValuesBucket {
  let obj: relationalStore.ValuesBucket = {};
  obj.modelFileUrl = modelPath.modelFileUrl;
  obj.modelName = modelPath.modelName;
  return obj;
}