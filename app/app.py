from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load the trained model, label encoders, and TF-IDF vectorizer
try:
    model = joblib.load("phishing_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load TF-IDF vectorizer
except FileNotFoundError:
    print("Error: Required model files not found. Ensure all are in the same directory.")
    exit()

# Define model features (including TF-IDF ones)
model_features = [
    "Having_@_symbol", "Having_IP", "Path", "Prefix_suffix_separation", "Protocol",
    "Redirection_//_symbol", "Sub_domains", "URL_Length", "age_domain", "dns_record",
    "domain_registration_length", "http_tokens", "statistical_report", "tiny_url",
    "web_traffic"
] + [f"domain_tfidf_{i}" for i in range(100)]  # 100 TF-IDF features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "Domain" not in data:
            return jsonify({'error': 'No URL provided'}), 400

        url = data["Domain"]
        input_df = pd.DataFrame([data])

        # Apply TF-IDF transformation to the domain
        tfidf_features = tfidf_vectorizer.transform([url]).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=[f"domain_tfidf_{i}" for i in range(100)])
        
        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    return jsonify({'error': f'Invalid value in column: {col}'}), 400
            elif col in model_features and col != 'label':
                input_df[col] = 0
        
        # Merge TF-IDF features into input
        input_df = input_df.drop(columns=['Domain'], errors='ignore')
        input_df = pd.concat([input_df, tfidf_df], axis=1)
        
        # Ensure all required features exist
        for c in set(model_features) - set(input_df.columns):
            input_df[c] = 0

        input_df = input_df[model_features].fillna(0)
        
        print(f"\n🔍 Processed Features for URL: {url}")
        print(input_df.to_string(index=False))
        
        prediction = model.predict(input_df)
        
        return jsonify({
            'url': url,
            'prediction': int(prediction[0]),  
            'label': 'Phishing' if prediction[0] == 1 else 'Legitimate'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
