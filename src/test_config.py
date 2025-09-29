"""Test configuration setup"""

from config import SPOTIFY_CLIENT_ID, CITIES, REGIONS, AUDIO_FEATURES

def test_config():
    """Test if configuration is loaded correctly"""
    
    print("Testing configuration...")
    
    # Test API credentials
    if SPOTIFY_CLIENT_ID:
        print(f"Spotify Client ID loaded: {SPOTIFY_CLIENT_ID[:8]}...")
    else:
        print("Spotify Client ID not found")
    
    # Test cities
    print(f"Loaded {len(CITIES)} cities")
    print(f"   First 5 cities: {CITIES[:5]}")
    
    # Test regions
    print(f"Loaded {len(REGIONS)} regions")
    print(f"   Regions: {list(REGIONS.keys())}")
    
    # Test audio features
    print(f"Loaded {len(AUDIO_FEATURES)} audio features")
    print(f"   Features: {AUDIO_FEATURES}")
    
    print("\n Configuration test complete!")

if __name__ == "__main__":
    test_config()