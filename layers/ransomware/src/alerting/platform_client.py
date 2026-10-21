"""
platform_client.py
---------------------
Sends alerts to the MAIN NEURASHIELD PLATFORM via HTTP POST, matching
the endpoint contract: POST /api/layer2/threat with a JSON body of
{"event_type": str, "severity": str}, and an X-API-Key header.

Security rules built in:
  - NO built-in API key. The key must come from the api_key argument or the
    NEURASHIELD_API_KEY environment variable. Without a key nothing is sent
    (and that is logged), instead of sending with a guessable default.
  - Only http:// and https:// URLs are accepted.
  - Redirects are NOT followed, so the API key can never be forwarded to a
    different server by a redirect.
  - If the platform URL is plain http:// to a non-local machine, a warning
    is logged, because the API key would travel unencrypted.
  - event_type and severity are checked before sending.

Failure handling: send_threat_alert() never raises. It returns a dict with
status "sent" or "failed", plus "retryable" (True for network problems and
server errors, False for configuration problems such as a rejected key).
The caller (AlertGenerator) keeps retryable failures in an outbox and tries
again later, so a short platform outage does not lose alerts.
"""

import os
from urllib.parse import urlparse

import requests

from src.utils.logger import get_logger, sanitize_for_log

logger = get_logger(__name__)

ALLOWED_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _shared_api_key() -> str:
    """The one NeuraShield platform key: env NEURASHIELD_API_KEY first, then the
    shared key file (NEURASHIELD_KEY_FILE) written by the supervisor."""
    k = (os.environ.get("NEURASHIELD_API_KEY") or "").strip()
    if k:
        return k
    path = os.environ.get("NEURASHIELD_KEY_FILE")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""
    return ""


class PlatformClient:
    def __init__(self, base_url: str, timeout_seconds: float = 5.0, api_key: str = None):
        parsed = urlparse(str(base_url).strip().rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"platform URL must start with http:// or https:// (got {sanitize_for_log(base_url)!r})")

        # Keep only scheme + host + port. send_threat_alert() adds the path.
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path:
            logger.warning(
                f"[platform_client] URL {sanitize_for_log(base_url)!r} included a path "
                f"({sanitize_for_log(parsed.path)!r}); using the origin only: {self.base_url!r}. "
                f"Set the platform URL to scheme+host+port only, e.g. 'http://localhost:8000'."
            )
        if parsed.scheme == "http" and (parsed.hostname or "").lower() not in _LOCAL_HOSTS:
            logger.warning("[platform_client] Platform URL is plain http:// to a non-local host: "
                           "the API key and alerts are NOT encrypted in transit. Use https://.")

        self.timeout_seconds = timeout_seconds
        self.api_key = api_key or _shared_api_key() or None
        self._warned_no_key = False

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def send_threat_alert(self, event_type: str, severity: str) -> dict:
        """
        Sends a threat alert to the platform's Layer 2 endpoint.
        Never raises. See the module docstring for the returned fields.
        """
        if not self.api_key:
            if not self._warned_no_key:
                logger.error("[platform_client] No API key configured (set NEURASHIELD_API_KEY). "
                             "Alerts are NOT being sent to the platform.")
                self._warned_no_key = True
            return {"status": "failed", "reason": "no_api_key", "retryable": False}

        severity = str(severity).upper()
        if severity not in ALLOWED_SEVERITIES:
            return {"status": "failed", "reason": "invalid_severity", "retryable": False}
        event_type = str(event_type)[:100]

        url = f"{self.base_url}/api/layer2/threat"
        payload = {"event_type": event_type, "severity": severity}

        try:
            response = requests.post(
                url, json=payload, headers={"X-API-Key": self.api_key},
                timeout=self.timeout_seconds, allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                logger.error(f"[platform_client] Platform answered with a redirect "
                             f"(HTTP {response.status_code}); not following it.")
                return {"status": "failed", "reason": "redirect_not_followed", "retryable": False}
            response.raise_for_status()
            try:
                result = response.json()
            except ValueError:
                result = {"raw": sanitize_for_log(response.text, 200)}
            logger.info(f"[platform_client] Alert sent: {url} | event_type={event_type!r} "
                        f"severity={severity!r} | platform response: {sanitize_for_log(result)}")
            return {"status": "sent", "platform_response": result, "retryable": False}

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[platform_client] Could not reach platform at {url}: {sanitize_for_log(e)}")
            return {"status": "failed", "reason": "connection_error", "detail": str(e)[:200], "retryable": True}
        except requests.exceptions.Timeout:
            logger.error(f"[platform_client] Platform request timed out after {self.timeout_seconds}s: {url}")
            return {"status": "failed", "reason": "timeout", "retryable": True}
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            detail = e.response.text[:200] if e.response is not None else str(e)
            logger.error(f"[platform_client] Platform rejected the request (HTTP {code}): "
                         f"{sanitize_for_log(detail, 200)}")
            # 401/403/422 = wrong key or wrong data: retrying will not help.
            # 5xx = platform problem: worth retrying.
            return {"status": "failed", "reason": "http_error", "http_status": code,
                    "detail": detail, "retryable": code >= 500}
        except requests.exceptions.RequestException as e:
            logger.error(f"[platform_client] Request failed: {sanitize_for_log(e)}")
            return {"status": "failed", "reason": "request_error", "retryable": False}


if __name__ == "__main__":
    # Run the mock first:  python -m src.alerting.mock_platform_server
    client = PlatformClient(base_url="http://localhost:9000", api_key="test-key-for-mock")
    print("Test 1 (CRITICAL):", client.send_threat_alert("ransomware_activity", "CRITICAL"))
    bad_client = PlatformClient(base_url="http://localhost:9999", timeout_seconds=2.0, api_key="x")
    result = bad_client.send_threat_alert("ransomware_activity", "HIGH")
    print("Test 2 (platform down):", result)
    assert result["status"] == "failed" and result["retryable"] is True
    print("PASSED: unreachable platform failed gracefully and is retryable.")
