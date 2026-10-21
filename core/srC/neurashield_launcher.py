# neurashield_launcher.py
# Entry point for the packaged Authentication Layer (build the .exe from
# THIS file, not from neurashield_platform_v3.py).
#
# Why a separate entry point:
#  * face_liveness.py runs the webcam window in a child process. In a
#    packaged build the child is the same .exe started with a mode flag
#    (--subprocess or --enroll-face). That flag must be handled BEFORE any
#    platform code is imported, because importing the platform loads the
#    model, starts the session monitor and opens the database.
#  * multiprocessing.freeze_support() is required for frozen Windows builds.
#
# Run from source:  python neurashield_launcher.py
# Environment:      NEURASHIELD_PORT (default 8000)
#                   NEURASHIELD_API_KEY, NEURASHIELD_DATA_DIR,
#                   NEURASHIELD_RESOURCE_DIR, NEURASHIELD_DEBUG_ENDPOINTS,
#                   NEURASHIELD_CORS_ORIGINS, NEURASHIELD_SESSION_LOCK

import multiprocessing
import os
import sys


def main():

    multiprocessing.freeze_support()

    if len(sys.argv) >= 2 and sys.argv[1] == "--show-key":

        from ns_paths import get_api_key

        print(get_api_key())

        return

    # Child-process modes first.
    if len(sys.argv) >= 2 and sys.argv[1] in ("--subprocess", "--enroll-face"):

        import face_liveness

        face_liveness.main_cli(sys.argv[1:])

        return

    import uvicorn

    from neurashield_platform_v3 import app, PORT

    # A windowed .exe has no console, so sys.stdout can be None and the
    # default uvicorn log formatter would fail. Use plain defaults then.
    log_config = None if sys.stdout is None else uvicorn.config.LOGGING_CONFIG

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        reload=False,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()