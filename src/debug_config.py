"""Debug configuration loading"""

import os
from dotenv import load_dotenv

print("🔍 Debugging configuration...")

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

# Try to load .env from different locations
print("\n📁 Looking for .env file...")

# Try current directory
if os.path.exists('.env'):
    print("✅ Found .env in current directory")
else:
    print("❌ No .env in current directory")

# Try parent directory (where it should be)
if os.path.exists('../.env'):
    print("✅ Found .env in parent directory")
    load_dotenv('../.env')
else:
    print("❌ No .env in parent directory")

# Check if variables are loaded
spotify_id = os.getenv('SPOTIFY_CLIENT_ID')
if spotify_id:
    print(f"✅ SPOTIFY_CLIENT_ID loaded: {spotify_id[:8]}...")
else:
    print("❌ SPOTIFY_CLIENT_ID not found")

spotify_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
if spotify_secret:
    print(f"✅ SPOTIFY_CLIENT_SECRET loaded: {spotify_secret[:8]}...")
else:
    print("❌ SPOTIFY_CLIENT_SECRET not found")