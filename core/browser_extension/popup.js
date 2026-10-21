chrome.storage.local.get("apiKey").then(({ apiKey }) => {
  const st = document.getElementById("status");
  if (apiKey) { st.textContent = "Key saved."; }
  else { st.textContent = "Key not set - context is not being sent."; }
});
document.getElementById("save").addEventListener("click", async () => {
  const v = document.getElementById("key").value.trim();
  await chrome.storage.local.set({ apiKey: v });
  document.getElementById("key").value = "";
  document.getElementById("status").textContent = v ? "Key saved." : "Key cleared.";
});
