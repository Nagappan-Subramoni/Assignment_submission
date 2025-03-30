from fastapi import FastAPI
import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bikeshare_model.processing import data_manager
from pydantic import BaseModel
from schemas.health import HealthResponse
import glob
import subprocess
from fastapi import APIRouter, BackgroundTasks
import signal
import psutil

# Create the main FastAPI app
app = FastAPI(
    title="Bike Share Prediction API",
    description="API for predicting bike rental demand based on various parameters",
    version="1.0.0",
    docs_url="/docs",  # Enables Swagger UI at /docs
    redoc_url="/redoc",  # Enables ReDoc at /redoc
    openapi_url="/openapi.json"  # The OpenAPI schema URL
)

# Create your router
router = APIRouter()


# Define the request model
class PredictionRequest(BaseModel):
    dteday: str
    season: str
    hr: str
    holiday: str
    weekday: str
    workingday: str
    weathersit: str
    temp: float
    atemp: float
    hum: int
    windspeed: float


class PredictionResponse(BaseModel):
    predicted_rentals: float


@router.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict bike rental demand based on input parameters.

    Parameters:
    - **dteday**: Date in YYYY-MM-DD format
    - **season**: Season (1:spring, 2:summer, 3:fall, 4:winter)
    - **hr**: Hour (0 to 23)
    - **holiday**: Whether day is holiday (0:No, 1:Yes)
    - **weekday**: Day of week (0 to 6)
    - **workingday**: Working day (0:No, 1:Yes)
    - **weathersit**: Weather situation (1:Clear, 2:Misty, 3:Light Snow/Rain, 4:Heavy Rain/Snow)
    - **temp**: Normalized temperature in Celsius
    - **atemp**: Normalized feeling temperature in Celsius
    - **hum**: Normalized humidity (0 to 100)
    - **windspeed**: Normalized wind speed

    Returns:
    - Predicted number of bike rentals
    """
    data = {'dteday': request.dteday, 'season': request.season, 'hr': request.hr,
            'holiday': request.holiday, 'weekday': request.weekday,
            'workingday': request.workingday, 'weathersit': request.weathersit,
            'temp': request.temp, 'atemp': request.atemp, 'hum': request.hum,
            'windspeed': request.windspeed}

    print(f'Got Input data {data}')
    X = data_manager.getUserDataPreprocessed(data)
    print(f'After preprocessing {X}')
    y_pred = data_manager.loadModelAndPredict(X_test=X)
    print(f'Predicted Value {y_pred}')
    return PredictionResponse(predicted_rentals=float(y_pred[0]))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(status="ok", message="API is running")


# Directory where .whl files are stored
WHL_DIRECTORY = os.getcwd()
print('Wheel Directory', WHL_DIRECTORY)


def get_latest_whl():
    """Find the latest .whl file in the directory."""
    whl_files = sorted(
        glob.glob(os.path.join(WHL_DIRECTORY, "*.whl")), key=os.path.getmtime, reverse=True
    )
    return whl_files[0] if whl_files else None


def install_latest_whl():
    """Installs the latest available .whl file."""
    latest_whl = get_latest_whl()
    if latest_whl:
        print(f"Installing: {latest_whl}")
        subprocess.run(["pip", "install", "--force-reinstall", "--upgrade", latest_whl], check=True)
        print("Upgrade complete. Restarting the server...")
        restart_server()
    else:
        print("No .whl files found.")


def restart_server():
    """Restarts the FastAPI server running with Uvicorn (Windows compatible)"""

    # Find and terminate the Uvicorn process
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any("uvicorn" in cmd for cmd in cmdline):
                print(f"Stopping Uvicorn process: {proc.info['pid']}")
                os.kill(proc.info['pid'], signal.SIGTERM)  # Terminate process
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Restart Uvicorn
    print("Restarting Uvicorn...")
    subprocess.Popen(
        ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


@router.get("/upgrade")
async def upgrade_service(background_tasks: BackgroundTasks):
    """Trigger a service upgrade by installing the latest .whl file"""
    background_tasks.add_task(install_latest_whl)
    return {"message": "Upgrade process started in the background."}


# Include the router in the main app
app.include_router(router)