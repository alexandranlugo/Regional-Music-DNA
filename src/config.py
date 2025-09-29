import os
from pathlib import Path
from dotenv import load_dotenv

# Get the directory containing this file, then go up one level
current_dir = Path(__file__).parent
project_root = current_dir.parent
env_path = project_root / '.env'

# Load environment variables
load_dotenv(env_path)

# API Configuration
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
LASTFM_API_KEY = os.getenv('LASTFM_API_KEY')

# Project Configuration
CITIES = [
    # West Coast
    'Los Angeles', 'San Francisco', 'Seattle', 'Portland',
    # South
    'Nashville', 'Austin', 'Atlanta', 'Miami', 'New Orleans',
    # Midwest
    'Chicago', 'Detroit', 'Minneapolis', 'Kansas City',
    # Northeast
    'New York', 'Boston', 'Philadelphia',
    # Mountain/Desert
    'Denver', 'Phoenix', 'Salt Lake City'
]

REGIONS = {
    'West Coast': ['Los Angeles', 'San Francisco', 'Seattle', 'Portland'],
    'South': ['Nashville', 'Austin', 'Atlanta', 'Miami', 'New Orleans'],
    'Midwest': ['Chicago', 'Detroit', 'Minneapolis', 'Kansas City'],
    'Northeast': ['New York', 'Boston', 'Philadelphia'],
    'Mountain/Desert': ['Denver', 'Phoenix', 'Salt Lake City']
}

# Audio features we'll analyze from Spotify
AUDIO_FEATURES = [
    'danceability',     # How suitable for dancing (0.0 to 1.0)
    'energy',          # Perceptual measure of intensity (0.0 to 1.0)
    'valence',         # Musical positivity/happiness (0.0 to 1.0)
    'acousticness',    # Whether track is acoustic (0.0 to 1.0)
    'instrumentalness', # Whether track has vocals (0.0 to 1.0)
    'speechiness',     # Presence of spoken words (0.0 to 1.0)
    'tempo',           # BPM (beats per minute)
    'loudness'         # Overall loudness in decibels
]

# Data collection settings
MAX_TRACKS_PER_CITY = 200
REQUEST_DELAY = 0.1  # Seconds between API requests