chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
    if (changeInfo.status == 'complete') {
      // You could send a message here to check the URL when the page is loaded.
      chrome.tabs.sendMessage(tabId, { action: "check-url" }, function(response) {
        if (response.label) {
          console.log(`Phishing Prediction: ${response.label}`);
        }
      });
    }
  });
  