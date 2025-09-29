import sys
import os

print("🐍 Python Environment Test")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Test imports
try:
    import pandas
    print("✅ Pandas imported successfully")
except ImportError:
    print("❌ Pandas not available")

try:
    import spotipy
    print("✅ Spotipy imported successfully")
except ImportError:
    print("❌ Spotipy not available")

try:
    from dotenv import load_dotenv
    print("✅ python-dotenv imported successfully")
except ImportError:
    print("❌ python-dotenv not available")

print("\n🎉 Environment test complete!")