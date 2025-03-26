import requests

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"

# Example URLs to test
test_url = "http://google.com/login"

# Loop through test URLs and send requests
response = requests.post(url, json={"url": test_url})

if response.status_code == 200:
    print(f"URL: {test_url} -> Prediction: {response.json()['prediction']}")
else:
    print(f"Error: {response.json()}")  # Print error if request fails
