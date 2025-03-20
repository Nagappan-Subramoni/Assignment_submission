import yaml
import pandas as pd
import bikeshare_model.pipeline as pipeline
import joblib
from bikeshare_model import DATASET_PATH,MODEL_PATH
    
def read_input_data():
    df=pd.read_csv(DATASET_PATH)
    return df

def fitAndSave(X_train,y_train):
    model= pipeline.piperline_fit.fit(X_train,y_train)
    joblib.dump(model,MODEL_PATH)
    return model

def loadModelAndPredict(X_test):
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    return y_pred

def getUserDataPreprocessed(data=None):
    if (data == None):
        data={'dteday': '05-11-2012' , 'season': 'winter' , 'hr' : '6am' , 'holiday' : 'No ','weekday' : 'Yes','workingday':'Yes', 'weathersit': 'Mist', 'temp': 6.1,'atemp' : 3.0014, 'hum': 49, 'windspeed': 10.0012}
        user=pd.DataFrame([data])
        user['dteday'] = pd.to_datetime('05-11-2012', format='%d-%m-%Y')
        user['registered']=None
        user['casual']=None
        user_y=129
        user=pipeline.piperline.fit_transform(user,[user_y])
        return user,user_y
    else:
        user=pd.DataFrame([data])
        user['dteday'] = pd.to_datetime('05-11-2012', format='%d-%m-%Y')
        user['registered']=None
        user['casual']=None
        user=pipeline.piperline.fit_transform(user)
        return user
