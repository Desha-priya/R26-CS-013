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

    // The platform requires the shared NeuraShield key (X-API-Key). It is
    // entered once in the extension popup and kept in extension storage.
    const { apiKey } = await chrome.storage.local.get("apiKey");
    if (!apiKey) {
      console.log("NeuraShield API key not set - open the extension popup.");
      return;
    }

    // Send to your NeuraShield platform
    const resp = await fetch("http://localhost:8000/api/context-update", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey
      },
      body: JSON.stringify(data)
    });

    if (resp.status === 401) {
      console.log("NeuraShield rejected the API key (401) - update it in the popup.");
      return;
    }
    console.log("Context sent:", data.url);
  } catch (err) {
    console.log("Could not send context (server may be offline):", err.message);
  }
}

// Send once when extension loads
sendCurrentTabInfo();