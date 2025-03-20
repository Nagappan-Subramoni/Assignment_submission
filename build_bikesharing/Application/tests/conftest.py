import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import pytest
from bikeshare_model import CONFIG_YML,MODEL_PATH,DATASET_PATH

@pytest.fixture
def model():
    assert os.path.exists(CONFIG_YML) , "config.yml file doesnt exist"
    assert os.path.exists(DATASET_PATH) , "train data missing"    
    assert os.path.exists(MODEL_PATH) , "Model is not found in expected location"
        



