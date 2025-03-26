import requests

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"

# Example URL to test
test_url = "https://www.betpawa.ug/events/popular?marketId=1X2&categoryId=2"

# Loop through test URLs and send requests
response = requests.post(url, json={"Domain": test_url})

if response.status_code == 200:
    print(f"URL: {test_url} -> Prediction: {response.json()['label']}")
else:
    print(f"Error: {response.json()}")  # Print error if request fails
