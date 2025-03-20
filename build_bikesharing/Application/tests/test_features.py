import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from sklearn.model_selection import train_test_split
from bikeshare_model.processing.data_manager import read_input_data
from bikeshare_model.processing.data_manager import loadModelAndPredict
from bikeshare_model.processing.data_manager import getUserDataPreprocessed 
from bikeshare_model.processing.data_manager import fitAndSave
import bikeshare_model.processing.data_manager as dm
from bikeshare_model import MODEL_PATH

import bikeshare_model.pipeline as pp

def test_read_input():
    df=read_input_data()
    expected_columns=['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday',
       'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual',
       'registered', 'cnt']
    assert list(df.columns) == expected_columns , " Columns dont match"

def test_valPipelines():
    df = read_input_data()
    X=df.drop(columns=['cnt'])
    y=df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = pp.piperline.fit_transform(X_train, y_train)
    X_test = pp.piperline.fit_transform(X_test, y_test)
    expected_columns=['season', 'hr', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp',
       'hum', 'windspeed', 'month', 'year', 'weekday_1', 'weekday_2',
       'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7']
    assert list(X_train.columns) == expected_columns , " Preprocessing Error"

def test_valFitSave():
    df = read_input_data()
    X=df.drop(columns=['cnt'])
    y=df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = pp.piperline.fit_transform(X_train, y_train)
    X_test = pp.piperline.fit_transform(X_test, y_test)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    model=fitAndSave(X_train,y_train)
    assert os.path.exists(MODEL_PATH) , "Model Not Saved Successfully"
    y_pred=loadModelAndPredict(X_test)
    assert len(y_pred) ==  len(y_test), " Mismatch in datasize"

test_read_input()
test_valPipelines()
test_valFitSave()