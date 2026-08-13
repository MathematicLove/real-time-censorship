import sys
import time
import argparse
import threading
import webbrowser
import urllib.request
import uvicorn
from app import config, APP_NAME, APP_VERSION
from app.logger import log

def wait_and_open(url):
    for _ in range(120):
        try:
            with urllib.request.urlopen(url + "/health", timeout=1) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.5)
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser(prog="real-time-censorship")
    parser.add_argument("mode", nargs="?", default="serve", choices=["serve", "live"])
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--device", default=config.FORCE_DEVICE, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    config.CAMERA_INDEX = args.camera
    config.FORCE_DEVICE = args.device
    config.API_HOST = args.host
    config.API_PORT = args.port

    url = "http://" + ("127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host) + ":" + str(args.port)
    if args.mode == "live":
        config.AUTO_CAMERA = True
        threading.Thread(target=wait_and_open, args=(url,), daemon=True).start()
    log(APP_NAME + " " + APP_VERSION + " starting, open " + url)
    settings = uvicorn.Config("app.api:api", host=args.host, port=args.port, log_level="warning", access_log=False)
    uvicorn.Server(settings).run()

    return 0

if __name__ == "__main__":
    sys.exit(main())