document.getElementById('check-url').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const url = tabs[0].url;
  
      document.getElementById('status').textContent = "Checking...";
  
      fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ Domain: url })
      })
      .then(response => response.json())
      .then(data => {
        const label = data.label;
        document.getElementById('status').textContent = "";
        document.getElementById('result').textContent = `${label}`;
      })
      .catch(error => {
        document.getElementById('status').textContent = "Error: " + error;
      });
    });
  });
  