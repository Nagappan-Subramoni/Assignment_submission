import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_YML=os.path.join(BASE_DIR,"config.yml")
DATASET_PATH=os.path.join(BASE_DIR,"datasets/bike-sharing-dataset.csv")
MODEL_PATH=os.path.join(BASE_DIR,"trained_models/model.pkl")
