import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import re

# Load datasets
legitimate_urls = pd.read_csv('data/legitimate-urls.csv')
phishing_urls = pd.read_csv('data/phishing-urls.csv')

# Assign labels
legitimate_urls["label"] = 0  # Legitimate
phishing_urls["label"] = 1  # Phishing

# Combine datasets
if 'Path' not in legitimate_urls.columns or 'Path' not in phishing_urls.columns:
    raise KeyError("Column 'Path' not found in dataset. Check column names.")

data = pd.concat([legitimate_urls, phishing_urls], ignore_index=True)

def extract_features(url):
    url = str(url)
    parsed_url = urlparse(url)
    return pd.Series({
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
    })

# Apply feature extraction
url_features = data['Path'].apply(extract_features)
data = pd.concat([data, url_features], axis=1)

# Extract features & target
X_text = data['Path'].fillna('')  # For TF-IDF
X_numeric = data.drop(['Path', 'label'], axis=1)  # Numeric features
y = data['label']

# Stratified sampling to maintain class balance
X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X_text, X_numeric, y, test_size=0.15, random_state=42, stratify=y
)

# Convert URLs into TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.95, min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

# Convert to DataFrame
X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), index=X_text_train.index)
X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), index=X_text_test.index)

# Ensure indices match before merging
X_train_combined = pd.concat([X_train_tfidf_df.reset_index(drop=True), X_numeric_train.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([X_test_tfidf_df.reset_index(drop=True), X_numeric_test.reset_index(drop=True)], axis=1)

# Ensure column names are strings
X_train_combined.columns = X_train_combined.columns.astype(str)
X_test_combined.columns = X_test_combined.columns.astype(str)

# Train Logistic Regression model with better cross-validation
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=10000)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_combined, y_train)

# Evaluate the model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
