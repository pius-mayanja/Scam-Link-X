import requests

url = "http://localhost:8000/predict"
data = {"url": "https://we.scam.com"}
response = requests.post(url, json=data)

print("Status Code:", response.status_code)  # Check HTTP status code
print("Raw Response:", response.text)       # See actual response content
