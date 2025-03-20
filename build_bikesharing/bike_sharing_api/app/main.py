import uvicorn
from api import app
import threading
import time
from fastapi import FastAPI
from api import install_latest_whl
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def auto_upgrade():
    print('Autoupgrade triggered')
    while True:
        install_latest_whl()
        time.sleep(3600)  # Check every hour (adjust as needed)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    app = FastAPI()
    # Include API routes
    app.include_router(app)
    # Run upgrade check in a background thread
    threading.Thread(target=auto_upgrade, daemon=True).start()
