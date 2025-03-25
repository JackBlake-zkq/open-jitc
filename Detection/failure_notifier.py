import sys
import requests

gpuid = sys.argv[1]

for line in sys.stdin:
    if "Segmentation fault" in line or "No devices were found" in line or "Unable to determine the device handle for" in line:
        print("Unrecoverable failure detected")
        # requests.post("http://localhost:5000/api/notify", json={"gpuid": gpuid, "recoverable": False})
    if "ECC error" in line:
        print("Recoverable failure detected")
        # requests.post("http://localhost:5000/api/notify", json={"gpuid": gpuid, "recoverable": True})