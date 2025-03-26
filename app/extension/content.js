chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.action == "check-url") {
      const url = window.location.href;
  
      fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ Domain: url })
      })
      .then(response => response.json())
      .then(data => {
        sendResponse({ label: data.label });
      })
      .catch(error => {
        sendResponse({ error: error });
      });
  
      // Indicate asynchronous response
      return true;
    }
  });
  