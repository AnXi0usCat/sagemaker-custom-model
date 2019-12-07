import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
import lightgbm as lgb
import pickle
import os
import sys
import json
import traceback

# default values for hyper parameters 
hyper_params_default = {
    'num_leaves': 12, 
    'objective': 'binary', 
    'metric': 'auc', 
    'seed': 7,
    'num_class': 2,
    'learning_rate': 0.01,
    "verbose": -1
}

# default values for training parameters
training_params_default = {
    'num_boost_round': 3000,
    'early_stopping_rounds': 10
}

# default types for hyper parameters 
hyper_params_default_types = {
    'num_leaves': int, 
    'objective': str, 
    'metric': str, 
    'seed': int,
    'num_class': int,
    'learning_rate': float,
    "verbose": int
}

# default types for training parameters
training_params_default_types = {
    'num_boost_round': int,
    'early_stopping_rounds': int
}

# paths where sagemaker does it's business
prefix = '/opt/ml/'
input_path =  os.path.join(prefix, 'input/data')
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')


# specify where we should get the data
training_channel_name = 'training'
validation_channel_name = 'validation'
training_path = os.path.join(input_path, training_channel_name)
validation_path = os.path.join(input_path, validation_channel_name)


def convert_to_dataset(df):
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return lgb.Dataset(x, label=y)


def load_data_from_files(paths, channel):
        input_files = [os.path.join(paths, file) for file in os.listdir(paths)]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(paths, channel))
        raw_data = [pd.read_csv(file, header=None) for file in input_files]
        data_df = pd.concat(raw_data)
        return data_df


def parse_parameters(params):
    # only work with copies so we dont override the truth
    hyper_params = hyper_params_default.copy()
    training_params = training_params_default.copy()
    
    for par in params:
        # if the hyper parameter is supported then we replace our default values
        # with provided value after converting it to the required type
        # becasue in SageMaker everyhting is passed as strings
        if par in hyper_params:
            hyper_params[par] = hyper_params_default_types[par](params[par])

        if par in training_params:
            training_params[par] = training_params_default_types[par](params[par])
  
    return hyper_params, training_params


def train():
    print('Starting training model: LightGBM')
    try:
        # get the hyperparam values
        with open(param_path, 'r') as tc:
            training_params = json.load(tc)
        hyper_p, training_p = parse_parameters(training_params)

        # training and validation data
        train = load_data_from_files(training_path, training_channel_name)
        valid = load_data_from_files(validation_path, validation_channel_name)
        
        # convert into format which lightgbm understands
        dtrain = convert_to_dataset(train)
        dvalid = convert_to_dataset(valid)
        
        model = lgb.train(hyper_p, 
                          dtrain, 
                          valid_sets=[dvalid],
                          num_boost_round=training_p['num_boost_round'], 
                          early_stopping_rounds=training_p['early_stopping_rounds'], 
                          verbose_eval=True)
        
        # save the trained model
        with open(os.path.join(model_path, 'light-gbm-model.pkl'), 'w') as out:
            pickle.dump(model, out)
        print('Training complete.')   
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    train()
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
