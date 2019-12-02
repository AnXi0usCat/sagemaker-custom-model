from rest_api import serve as srv
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
import lightgbm as lgb
import pickle

def convert_targets(row):
    if row.species == 'setosa':
        return 0
    elif row.species == 'versicolor':
        return 1
    elif row.species == 'virginica':
        return 2

def get_data_splits(dataframe, valid_fraction=0.1):
    valid_fraction = 0.1
    valid_size = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_size * 2]
    valid = dataframe[-valid_size * 2:-valid_size]
    test = dataframe[-valid_size:]
    return train, valid, test

def serve():
    # load model with pickle to predict
    with open('model.pkl', 'rb') as fin:
        pkl_bst = pickle.load(fin)

if __name__ == '__main__':
    iris = pd.read_csv('iris.csv')
    iris = shuffle(iris, random_state = 12345)
    iris['species'] = iris.apply(lambda x: convert_targets(x), axis=1)
    train, valid, test = get_data_splits(iris)

    feature_cols = train.columns.drop(['species'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['species'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['species'])

    param = {
        'num_leaves': 12, 
        'objective': 'multiclass', 
        'num_class': 3,
        'metric': 'multi_logloss', 
        'seed': 7, 
        'learning_rate': 0.01}
    print("Training model!")
    bst = lgb.train(param, dtrain, num_boost_round=3000, valid_sets=[dvalid], 
                    early_stopping_rounds=10, verbose_eval=False)

    valid_pred = bst.predict(valid[feature_cols])
    pred_class = [np.argmax(line) for line in valid_pred]
    accuracy = metrics.accuracy_score(pred_class, valid['species'])
    print('Validation set accuracy is: {}'.format(accuracy))

    print('saving the model as a pickle file')
    with open('lightgbm.pkl', 'wb') as fout:
        pickle.dump(bst, fout)

