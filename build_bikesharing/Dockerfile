# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /bike_sharing_api

# Copy the requirements file into the container
COPY bike_sharing_api/requirements.txt /bike_sharing_api/requirements.txt
COPY bike_sharing_api/bikesharingrentals-0.1.18-py3-none-any.whl /bike_sharing_api/bikesharingrentals-0.1.18-py3-none-any.whl

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /bike_sharing_api/requirements.txt

# Copy the FastAPI application code into the container
COPY bike_sharing_api/app /bike_sharing_api/app

# Command to run the application
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
