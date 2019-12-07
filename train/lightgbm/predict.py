import flask
import pandas as pd
import json
import os
import pickle
import lightgbm

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


class Predictor(object):
    model = None
       
    @classmethod
    def get_model(cls):
        """
        Load the model file from the model directory
        """
        if cls.model is None:
            with open(os.path.join(model_path, 'light-gbm-model.pkl'), 'rb') as input:
                cls.model = pickle.load(input)
        return cls.model
    
    @classmethod
    def predict(cls, input):
        """
        Make a prediction with the lightgbm model
        """
        model = cls.get_model()
        return model.predict(input)


app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine is the server is healthy
    """
    health = Predictor.get_model() is not None
    
    status = 200 if health else 500
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Convert the JSON input into a pandas dataframe and feed it to the model
    """
    data = None
    if flask.request.content_type == 'application/json':
        content = flask.request.json
        data = {
            'sepal_length': content['sepal_length'], 
            'sepal_width': content['sepal_width'],  
            'petal_length': content['petal_length'],
            'petal_width': content['petal_width']
        }
        output = Predictor.predict(pd.DataFrame(data, index=[0]))
        output = list(output[0])
        return flask.Response(
            response=json.dumps({'prediction': output}), 
            status=200, 
            mimetype='application/json'
        )
    else:
        return flask.Response(
            response=json.dumps({'message': 'invalid content type'}), 
            status=status, 
            mimetype='application/json'
        )
    
