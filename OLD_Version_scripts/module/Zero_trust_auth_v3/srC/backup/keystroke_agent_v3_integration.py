# ============================================================
# NeuraShield - Keystroke V3 Platform Integration
#
# Purpose:
#   Connect the validated Siamese BiLSTM V3 keystroke agent
#   to Risk Engine V3.
#
# IMPORTANT:
#   This module does NOT train the model.
#   This module does NOT modify model weights.
#   This module does NOT use the old behavioral models.
# ============================================================

from dataclasses import dataclass, asdict
from typing import Optional
import requests


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_PLATFORM_URL = "http://127.0.0.1:5000"

KEYSTROKE_DECISION_ENDPOINT = (
    "/api/platform/keystroke/decision"
)


# ============================================================
# REQUEST OBJECT
# ============================================================

@dataclass
class KeystrokeDecisionRequest:

    user_id: str
    session_id: str

    behavioral_similarity: float

    context: str = "normal_browsing"

    action: str = "VIEW_ACCOUNT"


# ============================================================
# CLIENT
# ============================================================

class KeystrokeV3PlatformClient:

    def __init__(
        self,
        platform_url=DEFAULT_PLATFORM_URL,
        timeout=10
    ):

        self.platform_url = (
            platform_url.rstrip("/")
        )

        self.timeout = timeout

    # ========================================================
    # SEND BEHAVIORAL DECISION
    # ========================================================

    def evaluate(
        self,
        user_id: str,
        session_id: str,
        behavioral_similarity: float,
        context: str = "normal_browsing",
        action: str = "VIEW_ACCOUNT"
    ):

        # ----------------------------------------------------
        # Safety validation
        # ----------------------------------------------------

        similarity = float(
            behavioral_similarity
        )

        if not 0.0 <= similarity <= 1.0:

            raise ValueError(
                "behavioral_similarity must "
                "be between 0 and 1"
            )

        request_data = (
            KeystrokeDecisionRequest(
                user_id=user_id,
                session_id=session_id,
                behavioral_similarity=similarity,
                context=context,
                action=action
            )
        )

        payload = asdict(
            request_data
        )

        # ----------------------------------------------------
        # Platform request
        # ----------------------------------------------------

        url = (
            self.platform_url
            + KEYSTROKE_DECISION_ENDPOINT
        )

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()

        result = response.json()

        return result


# ============================================================
# LOCAL INTEGRATION TEST
# ============================================================

def self_test():

    print("=" * 70)
    print("KEYSTROKE V3 PLATFORM INTEGRATION SELF TEST")
    print("=" * 70)

    client = (
        KeystrokeV3PlatformClient()
    )

    # --------------------------------------------------------
    # TEST 1
    # Normal legitimate user
    # --------------------------------------------------------

    print("\n[TEST 1] NORMAL USER")

    result = client.evaluate(

        user_id="USER-001",

        session_id="INTEGRATION-001",

        behavioral_similarity=0.94,

        context="normal_browsing",

        action="VIEW_PROFILE"
    )

    print(result)

    # --------------------------------------------------------
    # TEST 2
    # Behavioral anomaly
    # --------------------------------------------------------

    print("\n[TEST 2] BEHAVIORAL ANOMALY")

    result = client.evaluate(

        user_id="USER-001",

        session_id="INTEGRATION-002",

        behavioral_similarity=0.40,

        context="normal_browsing",

        action="VIEW_ACCOUNT"
    )

    print(result)

    # --------------------------------------------------------
    # TEST 3
    # Financial action
    # --------------------------------------------------------

    print("\n[TEST 3] FINANCIAL ACTION")

    result = client.evaluate(

        user_id="USER-001",

        session_id="INTEGRATION-003",

        behavioral_similarity=0.90,

        context="financial",

        action="TRANSFER_FUNDS"
    )

    print(result)

    print("\n" + "=" * 70)
    print("KEYSTROKE V3 PLATFORM INTEGRATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    self_test()