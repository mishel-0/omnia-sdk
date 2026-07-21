#!/usr/bin/env python3
"""
Generate a .omnia license key valid for 365 days.
Run this script to get a key. No account, no email, no cost.

Usage:
    python gen_key.py
    # Output: your-license-key

Then:
    mkdir -p ~/.omnia && echo "your-license-key" > ~/.omnia/license.key
"""
import json, time, base64, hashlib, hmac
from datetime import datetime

SECRET = b"omnia-2026-license-hmac-key-v1"

def generate(days: int = 365) -> str:
    expires = int(time.time()) + days * 86400
    payload = json.dumps({"expires": expires}, separators=(",", ":"))
    b64 = base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()
    sig = hmac.new(SECRET, b64.encode(), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"

if __name__ == "__main__":
    key = generate()
    info = json.loads(base64.urlsafe_b64decode(key.split(".")[0] + "=="))
    expires = datetime.fromtimestamp(info["expires"]).strftime("%Y-%m-%d")
    print(f".omnia License Key (expires {expires}):")
    print(key)
    print()
    print("Activate:")
    print(f"  mkdir -p ~/.omnia && echo '{key}' > ~/.omnia/license.key")
