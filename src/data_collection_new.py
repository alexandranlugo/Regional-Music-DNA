"""
Data collection module for Regional Music DNA project.
Handles comprehensive data gathering from Spotify, Last.fm, and Census APIs.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import time
import requests
import json
from datetime import datetime
import os

from config import (
    SPOTIFY_CLIENT_ID, 
    SPOTIFY_CLIENT_SECRET, 
    CENSUS_API_KEY,
    LASTFM_API_KEY,
    CITIES, 
    MAX_TRACKS_PER_CITY,
    REQUEST_DELAY
)

def initialize_spotify():
    """Initialize Spotify API client using Client Credentials Flow."""
    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        return sp
    except Exception as e:
        raise Exception(f"Failed to initialize Spotify client: {e}")

def test_spotify_connection():
    """Test if Spotify API credentials work."""
    try:
        sp = initialize_spotify()
        results = sp.search(q='test', type='track', limit=1)
        
        if results['tracks']['items']:
            track_name = results['tracks']['items'][0]['name']
            artist_name = results['tracks']['items'][0]['artists'][0]['name']
            print("Spotify API connection successful!")
            print(f"   Test track: '{track_name}' by {artist_name}")
            return True
        else:
            print("Spotify API returned empty results")
            return False
            
    except Exception as e:
        print(f"Spotify API connection failed: {e}")
        return False

def search_city_playlists(sp, city, limit=10):
    """Search for playlists related to a specific city."""
    try:
        print(f"Searching playlists for {city}...")
        
        search_queries = [
            f"{city} music",
            f"{city} playlist", 
            f"{city} hip hop",
            f"{city} indie",
            f"{city} local",
            f"best of {city}",
            f"{city} underground",
            f"{city} scene"
        ]
        
        all_playlists = []
        for query in search_queries:
            try:
                results = sp.search(q=query, type='playlist', limit=10)
                playlists = results['playlists']['items']
                
                for playlist in playlists:
                    if playlist and playlist.get('id'):
                        all_playlists.append({
                            'playlist_id': playlist['id'],
                            'name': playlist['name'],
                            'description': playlist.get('description', ''),
                            'owner': playlist['owner']['display_name'],
                            'tracks_total': playlist['tracks']['total'],
                            'search_query': query,
                            'city': city
                        })
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                print(f"Error with query '{query}': {e}")
                continue
        
        # Remove duplicates and filter by relevance
        unique_playlists = []
        seen_ids = set()
        
        for playlist in all_playlists:
            if playlist['playlist_id'] not in seen_ids:
                if (playlist['tracks_total'] > 10 and 
                    playlist['tracks_total'] < 1000 and 
                    city.lower() in playlist['name'].lower() + playlist['description'].lower()):
                    unique_playlists.append(playlist)
                    seen_ids.add(playlist['playlist_id'])
        
        print(f"Found {len(unique_playlists)} relevant playlists")
        return unique_playlists[:limit]
        
    except Exception as e:
        print(f"Error searching playlists for {city}: {e}")
        return []

def get_tracks_from_playlists(sp, playlists, max_tracks=100):
    """Extract tracks from playlists with metadata."""
    try:
        all_tracks = []
        tracks_collected = 0
        
        for playlist in playlists:
            if tracks_collected >= max_tracks:
                break
                
            try:
                playlist_id = playlist['playlist_id']
                playlist_name = playlist['name']
                
                results = sp.playlist_tracks(playlist_id, limit=50)
                tracks = results['items']
                
                while results['next'] and tracks_collected < max_tracks:
                    results = sp.next(results)
                    tracks.extend(results['items'])
                
                for item in tracks:
                    if tracks_collected >= max_tracks:
                        break
                        
                    track = item.get('track')
                    if track and track.get('id'):
                        track_info = {
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'artist_name': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                            'album_name': track['album']['name'],
                            'release_date': track['album'].get('release_date', ''),
                            'popularity': track.get('popularity', 0),
                            'duration_ms': track.get('duration_ms', 0),
                            'explicit': track.get('explicit', False),
                            'playlist_source': playlist_name,
                            'playlist_id': playlist_id,
                            'city': playlist['city']
                        }
                        all_tracks.append(track_info)
                        tracks_collected += 1
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                print(f"Error getting tracks from playlist {playlist.get('name', 'Unknown')}: {e}")
                continue
        
        print(f"   Collected {len(all_tracks)} tracks from playlists")
        return all_tracks
        
    except Exception as e:
        print(f"Error extracting tracks from playlists: {e}")
        return []

def collect_city_data(city, max_tracks=MAX_TRACKS_PER_CITY):
    """Collect comprehensive music data for a single city."""
    print(f"\n Collecting data for {city}")
    print("=" * 50)
    
    try:
        sp = initialize_spotify()
        
        playlists = search_city_playlists(sp, city, limit=15)
        if not playlists:
            print(f"No playlists found for {city}")
            return pd.DataFrame()
        
        tracks = get_tracks_from_playlists(sp, playlists, max_tracks)
        if not tracks:
            print(f"No tracks found for {city}")
            return pd.DataFrame()
        
        # Get additional track information instead of audio features
        track_ids = [track['track_id'] for track in tracks]
        enhanced_tracks = get_enhanced_track_info(sp, tracks)
        
        tracks_df = pd.DataFrame(enhanced_tracks)
        
        print(f"Successfully collected {len(tracks_df)} tracks for {city}")
        return tracks_df
            
    except Exception as e:
        print(f"Error collecting data for {city}: {e}")
        return pd.DataFrame()

def get_enhanced_track_info(sp, tracks):
    """Get additional track information to compensate for missing audio features."""
    enhanced_tracks = []
    
    for track in tracks:
        try:
            track_id = track['track_id']
            
            # Get detailed track info
            track_details = sp.track(track_id)
            
            # Extract artist info
            artist_id = track_details['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            
            # Enhance track data
            enhanced_track = track.copy()
            enhanced_track.update({
                'artist_genres': ', '.join(artist_info.get('genres', [])),
                'artist_popularity': artist_info.get('popularity', 0),
                'artist_followers': artist_info.get('followers', {}).get('total', 0),
                'track_popularity': track_details.get('popularity', 0),
                'preview_url': track_details.get('preview_url', ''),
                'explicit': track_details.get('explicit', False),
                'markets_available': len(track_details.get('available_markets', [])),
            })
            
            enhanced_tracks.append(enhanced_track)
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            print(f"Error enhancing track {track.get('track_name', 'Unknown')}: {e}")
            enhanced_tracks.append(track)  # Keep original if enhancement fails
            continue
    
    print(f"   Enhanced {len(enhanced_tracks)} tracks with additional metadata")
    return enhanced_tracks

def get_lastfm_city_data(city, limit=50):
    """Get top tracks/artists for a city from Last.fm API."""
    if not LASTFM_API_KEY:
        print("Last.fm API key not found, skipping Last.fm data")
        return None
        
    try:
        print(f"Getting Last.fm data for {city}...")
        
        base_url = "http://ws.audioscrobbler.com/2.0/"
        
        params = {
            'method': 'geo.gettoptracks',
            'country': 'United States',
            'location': city,
            'api_key': LASTFM_API_KEY,
            'format': 'json',
            'limit': limit
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'tracks' in data and 'track' in data['tracks']:
                tracks = data['tracks']['track']
                
                lastfm_data = {
                    'city': city,
                    'top_tracks': [],
                    'collection_date': datetime.now().isoformat()
                }
                
                for track in tracks:
                    track_info = {
                        'name': track.get('name', ''),
                        'artist': track.get('artist', {}).get('name', ''),
                        'playcount': int(track.get('playcount', 0)),
                        'listeners': int(track.get('listeners', 0)),
                        'url': track.get('url', '')
                    }
                    lastfm_data['top_tracks'].append(track_info)
                
                print(f"   Found {len(lastfm_data['top_tracks'])} tracks from Last.fm")
                return lastfm_data
            else:
                print(f"   No Last.fm data found for {city}")
                return None
        else:
            print(f"   Last.fm API error: {response.status_code}")
            return None
            
        time.sleep(REQUEST_DELAY)
        
    except Exception as e:
        print(f"Error getting Last.fm data for {city}: {e}")
        return None

def get_city_demographics(city):
    """Get demographic data for a city from Census API."""
    if not CENSUS_API_KEY:
        print("Census API key not found, skipping demographic data")
        return None
        
    try:
        print(f"Getting demographic data for {city}...")
        
        city_state_mapping = {
            'Los Angeles': 'CA', 'San Francisco': 'CA', 'Seattle': 'WA', 'Portland': 'OR',
            'Nashville': 'TN', 'Austin': 'TX', 'Atlanta': 'GA', 'Miami': 'FL', 'New Orleans': 'LA',
            'Chicago': 'IL', 'Detroit': 'MI', 'Minneapolis': 'MN', 'Kansas City': 'MO',
            'New York': 'NY', 'Boston': 'MA', 'Philadelphia': 'PA',
            'Denver': 'CO', 'Phoenix': 'AZ', 'Salt Lake City': 'UT'
        }
        
        state_fips = {
            'CA': '06', 'WA': '53', 'OR': '41', 'TN': '47', 'TX': '48', 'GA': '13', 
            'FL': '12', 'LA': '22', 'IL': '17', 'MI': '26', 'MN': '27', 'MO': '29',
            'NY': '36', 'MA': '25', 'PA': '42', 'CO': '08', 'AZ': '04', 'UT': '49'
        }
        
        city_fips = {
            'Los Angeles': '44000', 'San Francisco': '67000', 'Seattle': '63000', 'Portland': '59000',
            'Nashville': '52006', 'Austin': '05000', 'Atlanta': '04000', 'Miami': '45000', 'New Orleans': '55000',
            'Chicago': '14000', 'Detroit': '22000', 'Minneapolis': '43000', 'Kansas City': '38000',
            'New York': '51000', 'Boston': '07000', 'Philadelphia': '60000',
            'Denver': '20000', 'Phoenix': '55000', 'Salt Lake City': '67000'
        }
        
        if city not in city_state_mapping:
            print(f"   City {city} not in mapping, using placeholder data")
            return create_placeholder_demographics(city)
        
        state_code = state_fips.get(city_state_mapping[city])
        city_code = city_fips.get(city)
        
        if not state_code or not city_code:
            print(f"   FIPS codes not found for {city}, using placeholder data")
            return create_placeholder_demographics(city)
        
        base_url = "https://api.census.gov/data/2022/acs/acs5"
        
        variables = {
            'B01003_001E': 'total_population',
            'B25077_001E': 'median_home_value',
            'B19013_001E': 'median_household_income',
            'B01002_001E': 'median_age',
            'B15003_022E': 'bachelor_degree',
            'B15003_001E': 'total_education',
            'B02001_002E': 'white_alone',
            'B02001_003E': 'black_alone',
            'B02001_005E': 'asian_alone',
            'B03003_003E': 'hispanic_latino',
            'B02001_001E': 'total_race',
            'B25001_001E': 'total_housing_units',
            'B25003_002E': 'owner_occupied_housing',
            'B08301_010E': 'public_transportation',
            'B08301_001E': 'total_transportation'
        }
        
        var_string = ",".join(variables.keys())
        
        params = {
            'get': var_string,
            'for': f'place:{city_code}',
            'in': f'state:{state_code}',
            'key': CENSUS_API_KEY
        }
        
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if len(data) >= 2:
                header = data[0]
                values = data[1]
                
                raw_data = dict(zip(header, values))
                
                demographics = process_census_data(city, raw_data, variables)
                
                print(f"   Successfully collected demographic data for {city}")
                return demographics
            else:
                print(f"   Insufficient data returned for {city}")
                return create_placeholder_demographics(city)
        else:
            print(f"   Census API error {response.status_code} for {city}")
            return create_placeholder_demographics(city)
            
        time.sleep(REQUEST_DELAY)
        
    except Exception as e:
        print(f"Error getting demographic data for {city}: {e}")
        return create_placeholder_demographics(city)

def process_census_data(city, raw_data, variables):
    """Process raw Census API data into useful demographic metrics."""
    try:
        def safe_numeric(value, default=None):
            try:
                if value is None or value == '' or value == '-':
                    return default
                return float(value) if '.' in str(value) else int(value)
            except (ValueError, TypeError):
                return default
        
        total_pop = safe_numeric(raw_data.get('B01003_001E'))
        median_income = safe_numeric(raw_data.get('B19013_001E'))
        median_age = safe_numeric(raw_data.get('B01002_001E'))
        median_home_value = safe_numeric(raw_data.get('B25077_001E'))
        
        bachelor_degree = safe_numeric(raw_data.get('B15003_022E'), 0)
        total_education = safe_numeric(raw_data.get('B15003_001E'), 1)
        
        white_alone = safe_numeric(raw_data.get('B02001_002E'), 0)
        black_alone = safe_numeric(raw_data.get('B02001_003E'), 0)
        asian_alone = safe_numeric(raw_data.get('B02001_005E'), 0)
        hispanic_latino = safe_numeric(raw_data.get('B03003_003E'), 0)
        total_race = safe_numeric(raw_data.get('B02001_001E'), 1)
        
        owner_occupied = safe_numeric(raw_data.get('B25003_002E'), 0)
        total_housing = safe_numeric(raw_data.get('B25001_001E'), 1)
        
        public_transport = safe_numeric(raw_data.get('B08301_010E'), 0)
        total_transport = safe_numeric(raw_data.get('B08301_001E'), 1)
        
        demographics = {
            'city': city,
            'collection_date': datetime.now().isoformat(),
            'data_source': 'US Census ACS 5-Year 2022',
            
            'population': total_pop,
            'median_age': median_age,
            'median_household_income': median_income,
            'median_home_value': median_home_value,
            
            'education_bachelor_plus_percent': round((bachelor_degree / total_education) * 100, 2) if total_education > 0 else None,
            
            'race_white_percent': round((white_alone / total_race) * 100, 2) if total_race > 0 else None,
            'race_black_percent': round((black_alone / total_race) * 100, 2) if total_race > 0 else None,
            'race_asian_percent': round((asian_alone / total_race) * 100, 2) if total_race > 0 else None,
            'race_hispanic_percent': round((hispanic_latino / total_race) * 100, 2) if total_race > 0 else None,
            
            'homeownership_rate': round((owner_occupied / total_housing) * 100, 2) if total_housing > 0 else None,
            
            'public_transport_percent': round((public_transport / total_transport) * 100, 2) if total_transport > 0 else None,
            
            'urban_classification': classify_urban_type(total_pop),
            
            'raw_census_data': raw_data
        }
        
        return demographics
        
    except Exception as e:
        print(f"   Error processing Census data for {city}: {e}")
        return create_placeholder_demographics(city)

def classify_urban_type(population):
    """Classify city by population size."""
    if population is None:
        return 'Unknown'
    elif population >= 1000000:
        return 'Major City'
    elif population >= 300000:
        return 'Large City'
    elif population >= 100000:
        return 'Medium City'
    else:
        return 'Small City'

def create_placeholder_demographics(city):
    """Create placeholder demographic data when Census API fails."""
    return {
        'city': city,
        'collection_date': datetime.now().isoformat(),
        'data_source': 'Placeholder - Census API unavailable',
        'population': None,
        'median_age': None,
        'median_household_income': None,
        'median_home_value': None,
        'education_bachelor_plus_percent': None,
        'race_white_percent': None,
        'race_black_percent': None,
        'race_asian_percent': None,
        'race_hispanic_percent': None,
        'homeownership_rate': None,
        'public_transport_percent': None,
        'urban_classification': 'Unknown',
        'notes': 'Data collection failed - using placeholder'
    }

def collect_all_cities_data(cities=CITIES, save_raw=True):
    """Collect data for all cities and optionally save raw data."""
    print("Starting comprehensive data collection for all cities")
    print("=" * 60)
    
    all_data = {}
    lastfm_data = {}
    demographic_data = {}
    
    for i, city in enumerate(cities, 1):
        print(f"\n Processing city {i}/{len(cities)}: {city}")
        
        try:
            spotify_data = collect_city_data(city)
            all_data[city] = spotify_data
            
            city_lastfm = get_lastfm_city_data(city)
            if city_lastfm:
                lastfm_data[city] = city_lastfm
            
            city_demographics = get_city_demographics(city)
            if city_demographics:
                demographic_data[city] = city_demographics
            
            if save_raw and i % 5 == 0:
                save_data_checkpoint(all_data, lastfm_data, demographic_data, i)
            
        except Exception as e:
            print(f"Failed to collect data for {city}: {e}")
            continue
    
    if save_raw:
        save_final_data(all_data, lastfm_data, demographic_data)
    
    print(f"\n Data collection complete!")
    print(f"   Collected data for {len(all_data)} cities")
    
    return {
        'spotify_data': all_data,
        'lastfm_data': lastfm_data,
        'demographic_data': demographic_data
    }

def save_data_checkpoint(spotify_data, lastfm_data, demographic_data, checkpoint_num):
    """Save data checkpoint during collection."""
    try:
        os.makedirs('../data/raw', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for city, df in spotify_data.items():
            if not df.empty:
                filename = f"../data/raw/spotify_{city.replace(' ', '_').lower()}_{timestamp}.csv"
                df.to_csv(filename, index=False)
        
        if lastfm_data:
            filename = f"../data/raw/lastfm_data_checkpoint_{checkpoint_num}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(lastfm_data, f, indent=2)
        
        if demographic_data:
            filename = f"../data/raw/demographic_data_checkpoint_{checkpoint_num}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(demographic_data, f, indent=2)
        
        print(f"Checkpoint {checkpoint_num} saved")
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def save_final_data(spotify_data, lastfm_data, demographic_data):
    """Save final collected data."""
    try:
        os.makedirs('../data/raw', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_spotify_df = pd.concat(spotify_data.values(), ignore_index=True)
        spotify_filename = f"../data/raw/all_spotify_data_{timestamp}.csv"
        all_spotify_df.to_csv(spotify_filename, index=False)
        
        if lastfm_data:
            lastfm_filename = f"../data/raw/all_lastfm_data_{timestamp}.json"
            with open(lastfm_filename, 'w') as f:
                json.dump(lastfm_data, f, indent=2)
        
        if demographic_data:
            demo_filename = f"../data/raw/all_demographic_data_{timestamp}.json"
            with open(demo_filename, 'w') as f:
                json.dump(demographic_data, f, indent=2)
        
        print(f"Final data saved:")
        print(f"  Spotify: {spotify_filename}")
        print(f"  Last.fm: {lastfm_filename if lastfm_data else 'Not collected'}")
        print(f"  Demographics: {demo_filename if demographic_data else 'Not collected'}")
        
    except Exception as e:
        print(f"Error saving final data: {e}")

if __name__ == "__main__":
    print("🧪 Testing data collection functions...")
    
    # Test Spotify connection
    if not test_spotify_connection():
        print("Spotify connection failed")
        exit()
    
    test_city = "Nashville"
    print(f"\n Testing data collection for {test_city}...")
    
    sample_data = collect_city_data(test_city, max_tracks=10)
    if not sample_data.empty:
        print(f"Sample data shape: {sample_data.shape}")
        print(f"   Columns: {list(sample_data.columns)}")
        
        # Show what data we actually got
        if 'artist_genres' in sample_data.columns:
            print("Enhanced metadata successfully retrieved!")
        
    # Test other APIs
    lastfm_test = get_lastfm_city_data(test_city, limit=5)
    demo_test = get_city_demographics(test_city)
    
    print("\n Testing complete!")