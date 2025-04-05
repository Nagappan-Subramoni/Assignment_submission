import requests

url = "http://16.171.146.239:8000/predict/"
# url = "http://127.0.0.1:8000/predict/"

data = {
    "dteday": "05-11-2012",
    "season": "winter",
    "hr": "6am",
    "holiday": "No",
    "weekday": "Yes",
    "workingday": "Yes",
    "weathersit": "Mist",
    "temp": 6.1,
    "atemp": 3.0014,
    "hum": 49,
    "windspeed": 10.0012
}

response = requests.post(url, json=data)
print("Raw Response:", response.text)  # Print raw response before converting to JSON

if response.status_code == 200:
    print("Prediction response:", response.json())
else:
    print("Error:", response.status_code, response.text)
