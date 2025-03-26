from flask import Flask, request, jsonify
import pandas as pd
import joblib
import re
from urllib.parse import urlparse

model = joblib.load(r"C:\Users\mayan\Desktop\Scam-Link-X\phishing_model.pkl")
label_encoders = joblib.load(r"C:\Users\mayan\Desktop\Scam-Link-X\label_encoders.pkl")

app = Flask(__name__)

def preprocess_url(url):
    url = str(url)
    parsed_url = urlparse(url)

    # Feature extraction
    features = {
        'URL_Length': len(url),
        'tiny_url': 1 if len(url) < 15 else 0,
        'Protocol': 1 if parsed_url.scheme == 'https' else 0,
        'Redirection_//_symbol': 1 if '//' in url else 0,
        'Sub_domains': url.count('.') - 1,  # Subdomain count based on dots
        'Domain': parsed_url.netloc.split('.')[-2] if parsed_url.netloc else '',
        'age_domain': 1 if parsed_url.netloc.split('.')[-1] in ['com', 'net', 'org'] else 0,  # Simplified check
        'dns_record': 1 if parsed_url.netloc else 0,  # Just checking if there is a domain
        'domain_registration_length': len(parsed_url.netloc.split('.')[-1]) if parsed_url.netloc else 0,
        'http_tokens': 1 if 'http' in url else 0,  # Checking for "http" in the URL
        'Prefix_suffix_separation': 1 if '-' in url else 0,
        'Having_@_symbol': 1 if '@' in url else 0,
        'Having_IP': 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0,
        'Path': len(parsed_url.path),
        'statistical_report': 0,  # Dummy value, implement your logic if necessary
        'web_traffic': 0,  # Set a default value for web_traffic (you can improve this logic if needed)
    }

    # Convert to DataFrame for easier processing
    data = pd.DataFrame([features])

    # Reorder columns to match the model's expected input order
    expected_order = [
        'URL_Length', 'tiny_url', 'Protocol', 'Redirection_//_symbol', 'Sub_domains', 'Domain', 'age_domain', 
        'dns_record', 'domain_registration_length', 'http_tokens', 'Prefix_suffix_separation', 'Having_@_symbol', 
        'Having_IP', 'Path', 'statistical_report', 'web_traffic'
    ]
    data = data[expected_order]

    # Encode categorical features
    for column in data.select_dtypes(include=["object"]).columns:
        if column in label_encoders:
            try:
                data[column] = label_encoders[column].transform(data[column].astype(str))
            except ValueError:
                data[column] = 0  # Handle unseen categories by assigning 0

    # Fill missing values if necessary
    data = data.fillna(0)
    
    return data

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        url = data.get("url")
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        processed_data = preprocess_url(url)
        prediction = model.predict(processed_data)[0]
        return jsonify({"prediction": "phishing" if prediction == 1 else "legitimate"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
