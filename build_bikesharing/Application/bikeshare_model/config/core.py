import os
from dotenv import load_dotenv

# Load .env file (if exists)
load_dotenv()

class Config:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///default.db")
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create a global config object
config = Config()
