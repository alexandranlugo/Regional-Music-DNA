"""
Data cleaning module for regional music DNA project.
handles data quality issues and prepares data for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

def load_latest_data():
    """Load the most recent data files."""
    import os
    
    data_dir = '../data/raw/'
    
    #find latest Spotify file
    spotify_files = [f for f in os.listdir(data_dir) if f.startswith('all_spotify_data_') and f.endswith('.csv')]
    latest_spotify = sorted(spotify_files)[-1]
    
    print(f'Loading: {latest_spotify}')
    df = pd.read_csv(os.path.join(data_dir, latest_spotify))
    
    return df

def remove_duplicates(df):
    """Remove duplicate tracks."""
    print('\nREMOVING DUPLICATES')
    print('='*40)
    
    initial_count = len(df)
    print(f'Initial tracks: {initial_count:,}')
    
    #check diff types of duplicates
    track_id_dups = df.duplicated(subset=['track_id']).sum()
    track_artist_dups = df.duplicated(subset=['track_name', 'artist_name']).sum()
    
    print(f'Track ID duplicates: {track_id_dups}')
    print(f'Track+Artist duplicates: {track_artist_dups}')
    
    #remove exact track ID duplicates (keep first occurrence)
    df_clean = df.drop_duplicates(subset=['track_id'], keep='first')
    
    removed_count = initial_count - len(df_clean)
    print(f'Removed: {removed_count:,} duplicate tracks')
    print(f'Final tracks: {len(df_clean):,}')
    
    #check if removal affected city balance
    print(f'\nTracks per city after deduplication:')
    city_counts = df_clean['city'].value_counts()
    for city, count in city_counts.items():
        print(f'  {city}: {count}')
        
    return df_clean

def clean_genres(df):
    """Clean and standardize genre data."""
    print('\nCLEANING GENRES')
    print('='*40)
    
    initial_missing = df['artist_genres'].isnull().sum()
    print(f'Initial missing genres: {initial_missing} ({initial_missing / len(df) * 100:.1f}%)')
    
    #clean genre strings
    def clean_genre_string(genre_str):
        if pd.isna(genre_str) or genre_str == '':
            return None
        
        #split by comma and clean each genre
        genres = [g.strip().lower() for g in genre_str.split(',') if g.strip()]
        
        #remove empty or very short genres
        genres = [g for g in genres if len(g) > 2]
        
        #remove duplicates while preserving order
        seen = set()
        unique_genres = []
        for g in genres:
            if g not in seen:
                seen.add(g)
                unique_genres.append(g)

        return ', '.join(unique_genres) if unique_genres else None

    df['artist_genres_clean'] = df['artist_genres'].apply(clean_genre_string)
    
    final_missing = df['artist_genres_clean'].isnull().sum()
    print(f'Final missing genres: {final_missing} ({final_missing/len(df) * 100:.1f}%)')
    
    #show genre cleaning impact
    print(f'Genre cleaning impact:')
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        before = city_data['artist_genres'].notna().sum()
        after = city_data['artist_genres_clean'].notna().sum()
        print(f'  {city}: {before} -> {after} tracks with genres')
        
    return df

def clean_release_dates(df):
    """Clean and standardize release date data."""
    print('\nCLEANING RELEASE DATES')
    print('='*40)
    
    #convert release dates
    df['release_date_parsed'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    #extract components
    df['release_year'] = df['release_date_parsed'].dt.year
    df['release_month'] = df['release_date_parsed'].dt.month
    df['release_decade'] = (df['release_year'] // 10) * 10
    
    #check for invalid dates
    invalid_dates = df['release_date_parsed'].isnull().sum()
    future_dates = (df['release_year'] > 2025).sum()
    very_old_dates = (df['release_year'] < 1900).sum()
    
    print(f'Invalid dates: {invalid_dates}')
    print(f'Future dates (>2025): {future_dates}')
    print(f'Very old dates (<1900): {very_old_dates}')
    
    #show date range by city
    print(f'\nDate ranges by city:')
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        valid_years = city_data['release_year'].dropna()
        if len(valid_years) > 0:
            min_year = valid_years.min()
            max_year = valid_years.max()
            median_year = valid_years.median()
            print(f'  {city}: {min_year:0f}-{max_year:0f} (median: {median_year:0f})')
    return df

def handle_outliers(df):
    """Identify and handle outliers in numerical data."""
    print('\nHANDLING OUTLIERS')
    print('='*40)
    
    outlier_summary = {}
    
    #define outlier detection function
    def detect_outliers(series, method='iqr'):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers, lower_bound, upper_bound
    
    #check numerical columns
    numerical_cols = ['artist_popularity', 'track_popularity', 'artist_followers', 'duration_ms']
    
    for col in numerical_cols:
        if col in df.columns:
            outliers, lower, upper = detect_outliers(df[col])
            outlier_count = outliers.sum()
            
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'lower_bound': lower,
                'upper_bound': upper
            }
            
            print(f'{col}')
            print(f'  Outliers: {outlier_count} ({outlier_count / len(df) * 100:.1f}%)')
            print(f'  Valid range: {lower:.0f} - {upper:.0f}')
            
            #show extreme outliers
            extreme_outliers = df[outliers][col]
            if len(extreme_outliers) > 0:
                print(f'  Most extreme: {extreme_outliers.min():.0f}-{extreme_outliers.max():.0f}')
    
    for col in numerical_cols:
        if col in df.columns:
            outliers, _, _ = detect_outliers(df[col])
            df[f'{col}_is_outlier'] = outliers
    
    return df, outlier_summary

def create_derived_features(df):
    """Create useful derived features for analysis."""
    print('\nCREATING DERIVED FEATURES')
    print('='*40)
    
    #duration in minutes
    df['duration_minutes'] = df['duration_ms'] / 60000
    
    #artist follower tiers
    df['artist_tier'] = pd.cut(df['artist_followers'],
                               bins=[0,1000,10000,100000,1000000,float('inf')],
                               labels=['Emerging','Small','Medium','Large','Superstar'])
    
    #popularity categories
    df['popularity_category'] = pd.cut(df['track_popularity'],
                                       bins=[0,20,40,60,80,100],
                                       labels=['Unknown','Low','Medium','High','Viral'])
    
    #music era based on released decade
    def categorize_era(decade):
        if pd.isna(decade):
            return 'Unknown'
        elif decade < 1980:
            return 'Classic'
        elif decade < 2000:
            return 'Vintage'
        elif decade < 2010:
            return '2000s'
        elif decade < 2020:
            return '2010s'
        else:
            return 'Current'
    
    df['music_era'] = df['release_decade'].apply(categorize_era)
    
    #genre count per track
    df['genre_count'] = df['artist_genres_clean'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
    
    #track age in years
    current_year = datetime.now().year
    df['track_age_years'] = current_year - df['release_year']
    
    print(f'Created derived features:')
    print(f'  - duration_minutes')
    print("   - artist_tier")
    print("   - popularity_category")
    print("   - music_era")
    print("   - genre_count")
    print("   - track_age_years")
    
    return df

def validated_cleaned_data(df):
    """Validate the cleaned dataset."""
    print('\nVALIDATING CLEANED DATA')
    print('='*40)
    
    validation_results = {
        'total_tracks': len(df),
        'cities': df['city'].nunique(),
        'tracks_per_city': df['city'].value_counts().to_dict(),
        'data_completeness': {},
        'issues': []
    }
    
    #check data completeness
    critical_columns = ['track_id', 'track_name', 'artist_name', 'city']
    for col in critical_columns:
        completeness = df[col].notna().sum() / len(df) * 100
        validation_results['data_completeness'][col] = completeness
        if completeness < 100:
            validation_results['issues'].append(f'Missing {col}: {100 - completeness:.1f}%')
            
    #check city balance
    city_counts = df['city'].value_counts()
    if city_counts.std()/city_counts.mean() > 0.1:
        validation_results['issues'].append('Uneven city distribution')
                
    #print validation summary
    print(f"Final dataset: {validation_results['total_tracks']:,} tracks")
    print(f"Cities: {validation_results['cities']}")
            
    print(f'\n Tracks per city:')
    for city, count in validation_results['tracks_per_city'].items():
        print(f'  {city}: {count}')
                
    if validation_results['issues']:
        print(f'\nIssues found:')
        for issue in validation_results['issues']:
            print(f'  - {issue}')
    else:
        print(f'\nNo major issues found!')
                
    return validation_results  

def clean_full_dataset():
    """Main function clean the entire dataset."""
    print("REGIONAL MUSIC DNA - DATA CLEANING")
    print("=" * 60)
    
    #load data
    df = load_latest_data()
    print(f'Loaded {len(df):,} tracks from {df["city"].nunique()} cities')
    
    #apply cleaning steps
    df = remove_duplicates(df)
    df = clean_genres(df)
    df = clean_release_dates(df)
    df, outlier_summary = handle_outliers(df)
    df = create_derived_features(df)
    
    #validate results
    validation_results = validated_cleaned_data(df)
    
    #save cleaned data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'../data/processed/spotify_cleaned_{timestamp}.csv'
    
    #create processed dierectory if it doesn't exist
    import os
    os.makedirs('../data/processed', exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f'\nCleaned data saved to: {output_file}')
    
    return df, validation_results, outlier_summary

if __name__ == "__main__":
    cleaned_df, validation_results, outlier_summary = clean_full_dataset()
    
    print("\nData cleaning complete!")
    print(f'Ready for analysis with {len(cleaned_df):,} cleaned tracks!')