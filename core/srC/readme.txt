Here all script for main sytem (not for model- models related store at scripts)


Context-aware adaptive continuous authentication in a Zero Trust architecture, where behavioral-biometric 
confidence is combined with cross-layer security alerts and transaction/context sensitivity to dynamically 
change the authentication threshold.


#neurashieldPlatform architecture
#   KeystrokeAgentV3
#          |
#          | behavioral evidence
#          v
#   NeuraShield Platform
#          |
#          +---- Session Context
#          |
#          +---- Cross-Layer Alerts
#          |
#          +---- Financial Context
#          |
#          v
#   RiskEngineV3
#          |
#          v
#   ALLOW / MONITOR / REAUTHENTICATE
#
# Old IF/SVM behavioral models:
#     DISABLED
# ============================================================
# CORS
#
# No cross-origin browser traffic currently needs to reach this
# platform directly - her Layer 4 and Layer 2 dashboards each talk to
# their own backend (ports 8001 / 5050), which forwards to this
# platform server-side, not from the browser. This allow-list is
# deliberately narrow (only this system's own local dashboard origins,
# not "*") so a browser-based request from anywhere else is rejected
# by the browser itself, even if it somehow obtained a valid API key.
# Extend this list explicitly if a future dashboard needs direct
# browser-to-platform calls - do not widen it to "*".
# ============================================================

#app.add_middleware 

# ============================================================
# RATE LIMITING
#
# A deliberately simple, dependency-free, in-process rate limiter -
# not a production-grade distributed limiter (no external cache like
# Redis, resets if the process restarts, per-process not per-cluster).
# For a single-process, single-machine, on-prem platform like this,
# that's an honest, sufficient tradeoff - the goal is to stop one
# client (a buggy layer, a runaway loop, a local actor) from hammering
# the API, not to defend a multi-instance internet-facing service.
#
# Sliding/fixed window per client IP: each IP gets RATE_LIMIT_MAX
# requests per RATE_LIMIT_WINDOW_SECONDS; requests over that get a 429
# instead of being processed. Static/docs routes are exempt since
# browser polling for those isn't a meaningful abuse vector.
# ============================================================

RATE_LIMIT_WINDOW_SECONDS = 10
RATE_LIMIT_MAX_REQUESTS = 60  # 6 req/sec sustained - generous for normal dashboard polling
#-
if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            # NOTE: raising HTTPException directly from @app.middleware
            # is not reliably converted into a clean JSON response by
            # FastAPI's exception handlers (that mechanism is built for
            # route/dependency-level exceptions) - it can surface as an
            # unhandled 500 instead of the intended 429. Returning a
            # JSONResponse directly here is the reliable way to short-
            # circuit a request from inside HTTP middleware.
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} "
                        f"requests per {RATE_LIMIT_WINDOW_SECONDS}s. Try again shortly."
                    )
                },
            )
 # ============================================================
# API AUTHENTICATION
#
# Shared-secret API key for a single-operator local security
# console. This is NOT per-user auth or enterprise SSO — it is
# the minimum viable control to stop an unauthenticated actor
# on the network from injecting fake alerts, resetting the
# enrolled biometric profile, or forging risk decisions.
#
# Set NEURASHIELD_API_KEY before starting the platform in any
# non-throwaway environment. The default below is for local
# development only.
# ============================================================

NEURASHIELD_API_KEY = os.environ.get(           
#-----------------
#pin code i set = 456123   = 

#  rsk engine v3
# Behavioral source:
#     Siamese BiLSTM V3 cosine similarity
#
# Responsibilities:
#     - behavioral biometric risk
#     - contextual risk
#     - cross-layer alerts
#     - adaptive authentication threshold
#     - action sensitivity
#     - continuous authentication decisions
#
# IMPORTANT:
#     No training.
#     No weight updates.
#     No Isolation Forest.
#     No One-Class SVM.
#     No old behavioral models.
# ============================================================




    # Number of consecutive REAUTHENTICATE-tier windows required before actually forcing step-up auth. A single noisy/unlucky
    # window escalates to MONITOR instead of interrupting the session outright — reduces false step-ups from one-off
    # typing noise while still catching sustained anomalies fast.

    # This is now CONTEXT-DEPENDENT: normal browsing tolerates more sustained anomaly before forcing a challenge (fewer
    #       interruptions for everyday use), while sensitive/financial/
    #       attack contexts keep the original, stricter requirement —
    #       deliberately reacting faster exactly where a false accept
    #       would be most costly. The underlying similarity threshold
    #       itself (VALIDATED_SIMILARITY_THRESHOLD) is UNCHANGED — this
    #       only tunes how much sustained evidence is required before
    #       interrupting the session, not how "wrong" a single window
    #       is allowed to look.

CONSECUTIVE_REAUTH_REQUIRED = 2  # default / sensitive contexts

 # After a successful, identity-confirmed step-up, suppress re-triggering REAUTHENTICATE for this long in NON-sensitive
    # contexts (normal browsing). Without this, a user whose typing sits persistently near the threshold gets challenged
    # every ~10 seconds forever. Grace is withdrawn the moment context_risk >= 0.30 (sensitive/financial/attack).
    VERIFIED_GRACE_SECONDS = 120


      # ----------------------------------------------------
        # POST-VERIFICATION GRACE PERIOD
        #
        # A recently passed step-up (this user) suppresses
        # re-escalating to REAUTHENTICATE in normal contexts —
        # downgraded to MONITOR instead. Never applies once
        # context_risk >= 0.30 (sensitive/financial/attack),
        # where full scrutiny is always re-enforced.
        # ----------------------------------------------------

        grace_key = (
            str(user_id)
            if user_id is not None
            else "global"
        )

          # ----------------------------------------------------
        # SMOOTHING
        #
        # Only escalate to REAUTHENTICATE after
        # CONSECUTIVE_REAUTH_REQUIRED consecutive windows past
        # the reauth boundary for this user/session. A single
        # dip still shows up as MONITOR (visible, logged) but
        # does not interrupt the session by itself.
        # ----------------------------------------------------

        smoothing_key = (
            str(user_id)
            if user_id is not None
            else "global"
        )