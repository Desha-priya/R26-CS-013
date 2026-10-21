// background.js
console.log("NeuraShield Context Detector started");

// Listen for tab changes
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  sendCurrentTabInfo();
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") {
    sendCurrentTabInfo();
  }
});

async function sendCurrentTabInfo() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || !tab.url || tab.url.startsWith("chrome://")) return;

    const data = {
      url: tab.url,
      title: tab.title || "",
      timestamp: new Date().toISOString()
    };

    // Send to your NeuraShield platform
    await fetch("http://localhost:8000/api/context-update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    console.log("Context sent:", data.url);
  } catch (err) {
    console.log("Could not send context (server may be offline):", err.message);
  }
}

// Send once when extension loads
sendCurrentTabInfo();