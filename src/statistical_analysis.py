"""
Statistical analysis module for Regional Music DNA project.
Performs comprehensive statistical analysis on cleaned music data.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Loads most recent cleaned dataset."""
    import os
    
    processed_dir = '../data/processed/'
    cleaned_files = [f for f in os.listdir(processed_dir) if f.startswith('spotify_cleaned_') and f.endswith('.csv')]
    
    if not cleaned_files:
        raise FileNotFoundError('No cleaned data files found. Run data cleaning first.')
    
    latest_file = sorted(cleaned_files)[-1]
    df = pd.read_csv(os.path.join(processed_dir, latest_file))
    
    print(f"Loaded cleaned data: {latest_file}")
    print(f"Dataset: {len(df):,} tracks from {df['city'].nunique()} cities")
    
    return df

def descriptive_statistics_by_city(df):
    """Calculate comprehensive descriptive statistics by city."""
    print('\nDESCRIPTIVE STATISTICS BY CITY')
    print('='*50)
    
    #numerical variables to analyze
    numerical_vars = ['artist_popularity','track_popularity','artist_followers',
                      'duration_minutes','genre_count','track_age_years']
    
    stats_summary = {}
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        city_stats = {}
        
        print('\n{city} (n={len(city_data)})')
        print('-'*30)
        
        for var in numerical_vars:
            if var in df.columns:
                data = city_data[var].dropna()
                
                if len(data) > 0:
                    city_stats[var] = {
                        'mean': data.mean(),
                        'median': data.median(),
                        'std': data.std(),
                        'min': data.min(),
                        'max': data.max(),
                        'q25': data.quantile(0.25),
                        'q75': data.quantile(0.75),
                        'skewness': stats.skew(data),
                        'kurtosis': stats.kurtosis(data)
                    }
                    
                    print(f"   {var}:")
                    print(f"      Mean: {city_stats[var]['mean']:.2f}")
                    print(f"      Median: {city_stats[var]['median']:.2f}")
                    print(f"      Std: {city_stats[var]['std']:.2f}")
                    print(f"      Range: {city_stats[var]['min']:.1f} - {city_stats[var]['max']:.1f}")
        
        stats_summary[city] = city_stats
    
    return stats_summary

def genre_analysis_advanced(df):
    """Advanced statistical analysis of genre patterns"""
    print('\nADVANCED GENRE ANALYSIS')
    print('='*50)
    
    genre_analysis = {}
    
    #create genre frequency matrix
    genre_city_matrix = {}
    all_genres = set()
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        city_genres = []
        
        for genres_str in city_data['artist_genres_clean'].dropna():
            if genres_str:
                genres = [g.strip() for g in genres_str.split(',')]
                city_genres.extend(genres)
                all_genres.update(genres)
                
        #calculate genre frequencies
        genre_counts = pd.Series(city_genres).value_counts()
        total_tracks = len(city_data)
        genre_frequencies = (genre_counts / total_tracks * 100).round(2)
        
        genre_city_matrix[city] = genre_frequencies
        
        #store analysis results
        genre_analysis[city] = {
            'total_tracks': total_tracks,
            'tracks_with_genres': city_data['artist_genres_clean'].notna().sum(),
            'unique_genres': len(genre_counts),
            'top_genre': genre_frequencies.index[0] if len(genre_frequencies) > 0 else None,
            'top_genre_pct': genre_frequencies.iloc[0] if len(genre_frequencies) > 0 else None,
            'genre_diversity': len(genre_counts) / total_tracks if total_tracks > 0 else None,
            'top_5_genres': genre_frequencies.head(5).to_dict()
        }
    
    #create genre city matrix for statistical test
    genre_df = pd.DataFrame(genre_city_matrix).fillna(0)
    
    #calculate genre dominance scores
    print('\nGenre Dominance Analysis:')
    for city in df['city'].unique():
        analysis = genre_analysis[city]
        print(f"\n{city}:")
        print(f"  Unique genres: {analysis['unique_genres']}")
        print(f"  Genre diversity: {analysis['genre_diversity']:.3f}")
        print(f"  Top genre: {analysis['top_genre']} ({analysis['top_genre_pct']:.1f}%)")
        
        #show top 3 genres
        top_genres = analysis['top_5_genres']
        for i, (genre, pct) in enumerate(list(top_genres.items())[:3],1):
            print(f'  {i}. {genre}: {pct:.1f}%')
    
    return genre_analysis, genre_df

def popularity_analysis(df):
    """Statistical analysis of popularity patterns."""
    print('\nPOPULARITY ANALYSIS')
    print('='*50)
    
    popularity_results={}
    
    #test for diff in popularity across cities
    cities = df['city'].unique()
    
    #artist pop analysis
    artist_pop_groups = [df[df['city'] == city]['artist_popularity'].dropna() for city in cities]
    
    #kruskal-wallis test (non-parametric ANOVA)
    if len(artist_pop_groups) > 2:
        h_stat, p_value = kruskal(*artist_pop_groups)
        
        print(f'Artist Popularity Differences Across Cities')
        print(f'  Kruskal-Wallis H-statistic: {h_stat:.3f}')
        print(f'  p-value: {p_value:.6f}')
        print(f'  Significant difference: {"Yes" if p_value < 0.05 else "No"}')
        
        popularity_results['artist_popularity_test'] = {
            'h_stat': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    #track pop analysis
    track_pop_groups = [df[df['city'] == city]['track_popularity'].dropna() for city in cities]
    
    
    if len(track_pop_groups) > 2:
        h_stat, p_value = kruskal(*track_pop_groups)
        
        print(f'Track Popularity Differences Across Cities')
        print(f'  Kruskal-Wallis H-statistic: {h_stat:.3f}')
        print(f'  p-value: {p_value:.6f}')
        print(f'  Significant difference: {"Yes" if p_value < 0.05 else "No"}')
        
        popularity_results['track_popularity_test'] = {
            'h_stat': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
    #pop tier analysis
    print(f'\nPopularity Distribution by City:')
    popularity_crosstab = pd.crosstab(df['city'], df['popularity_category'])
    
    for city in cities:
        city_data = df[df['city'] == city]
        tier_dist = city_data['popularity_category'].value_counts(normalize=True) * 100
        
        print(f'\n{city}:')
        for tier, pct in tier_dist.items():
            print(f'  {tier}: {pct:.1f}%')
            
    #chi-square test for popularity distribution
    chi2, p_val, dof, expected = chi2_contingency(popularity_crosstab)
    
    print(f'\nPopularity Distribution Independence Test:')
    print(f'  Chi-square statistic: {chi2:.3f}')
    print(f'  p-value: {p_val:.6f}')
    print(f'  Degrees of Freedom: {dof}')
    print(f'  Distributions indepedent: {"No" if p_val < 0.05 else "Yes"}')
    
    popularity_results['distribution_test'] = {
        'chi2_statistic': chi2,
        'p_value': p_val,
        'degrees_of_freedom': dof,
        'independent': p_val >= 0.05
    }
    
    return popularity_results

def temporal_analysis(df):
    """Analyze temporal patterns in music preferences."""
    print("\nTEMPORAL PATTERN ANALYSIS")
    print("=" * 50)
    
    temporal_results = {}
    
    # Music era distribution by city
    print("Music Era Distribution by City:")
    era_crosstab = pd.crosstab(df['city'], df['music_era'])
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        era_dist = city_data['music_era'].value_counts(normalize=True) * 100
        
        print(f"\n{city}:")
        for era, pct in era_dist.items():
            print(f"  {era}: {pct:.1f}%")
    
    # Test for independence of era distribution
    chi2, p_val, dof, expected = chi2_contingency(era_crosstab)
    
    print(f"\nMusic Era Distribution Independence Test:")
    print(f"  Chi-square statistic: {chi2:.3f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Era preferences independent of city: {'No' if p_val < 0.05 else 'Yes'}")
    
    # Average track age by city
    print(f"\nAverage Track Age by City:")
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        avg_age = city_data['track_age_years'].mean()
        median_age = city_data['track_age_years'].median()
        
        print(f"  {city}: {avg_age:.1f} years (median: {median_age:.1f})")
    
    temporal_results = {
        'era_distribution_test': {
            'chi2_statistic': chi2,
            'p_value': p_val,
            'independent': p_val >= 0.05
        }
    }
    
    return temporal_results


def artist_tier_analysis(df):
    """Analyze artist tier distributions across cities."""
    print("\nARTIST TIER ANALYSIS")
    print("=" * 50)
    
    # Artist tier distribution
    tier_crosstab = pd.crosstab(df['city'], df['artist_tier'])
    
    print("Artist Tier Distribution by City:")
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        tier_dist = city_data['artist_tier'].value_counts(normalize=True) * 100
        
        print(f"\n{city}:")
        for tier, pct in tier_dist.items():
            print(f"   {tier}: {pct:.1f}%")
    
    # Test for independence
    chi2, p_val, dof, expected = chi2_contingency(tier_crosstab)
    
    print(f"\nArtist Tier Distribution Independence Test:")
    print(f"  Chi-square statistic: {chi2:.3f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Tier preferences independent of city: {'No' if p_val < 0.05 else 'Yes'}")
    
    return {
        'tier_distribution_test': {
            'chi2_statistic': chi2,
            'p_value': p_val,
            'independent': p_val >= 0.05
        }
    }
    
def correlation_analysis(df):
    """Analyze correlations between numerical variables."""
    print('\nCORRELATION ANALYSIS')
    print('='*50)
    
    #numerical variables to analyze
    numerical_cols = ['artist_popularity','track_popularity','artist_followers',
                      'duration_minutes','genre_count','track_age_years']
    
    #filter to available columns
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print('Insufficient numerical columns for correlation analysis')
        return {}
    
    correlation_matrix = df[available_cols].corr()
    
    print('Correlation Matrix:')
    print(correlation_matrix.round(3))
    
    #find strongest correlations
    print(f'\nStrongest Correlations:')
    correlations = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i,j]
            
            correlations.append({
                'variable1': var1,
                'variable2': var2,
                'correlation': corr_value,
                'abs_correlation': abs(corr_value)
            })
    
    #sort by absolute correlation value
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    for corr in correlations[:5]: #top 5 correlations
        strength = 'Strong' if corr['abs_correlation'] > 0.7 else 'Moderate' if corr['abs_correlation'] > 0.3 else 'Weak'
        direction = 'positive' if corr['correlation'] > 0 else 'negative'
        
        print(f"  {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f} ({strength} {direction})")
        
    return {
        'correlation_matrix': correlation_matrix,
        'top_correlations': correlations[:10]
    }
    
def city_similarity_analysis(df):
    """Analyze similarity between cities using multiple metrics."""
    print('\nCITY SIMILARITY ANALYSIS')
    print('='*50)
    
    cities = df['city'].unique()
    similarity_results = {}
    
    #create city profiles for comparison
    city_profiles = {}
    
    for city in cities:
        city_data = df[df['city'] == city]
        
        profile = {
            'avg_artist_popularity': city_data['artist_popularity'].mean(),
            'avg_track_popularity': city_data['track_popularity'].mean(),
            'avg_followers': city_data['artist_followers'].mean(),
            'avg_duration': city_data['duration_minutes'].mean(),
            'avg_genre_count': city_data['genre_count'].mean(),
            'avg_track_age': city_data['track_age_years'].mean(),
            'explicit_rate': city_data['explicit'].mean(),
        }
        
        city_profiles[city] = profile
    
    #convert to dataframe for easier analysis
    profile_df = pd.DataFrame(city_profiles).T
    
    #standardize the features
    scaler = StandardScaler()
    profile_scaled = scaler.fit_transform(profile_df)
    profile_scaled_df = pd.DataFrame(profile_scaled,
                                     index=profile_df.index,
                                     columns=profile_df.columns)
    
    #calculate city distances (euclidean distance)
    from scipy.spatial.distance import pdist, squareform
    
    distances = pdist(profile_scaled, metric='euclidean')
    distance_matrix = squareform(distances)
    distance_df = pd.DataFrame(distance_matrix, index=cities, columns=cities)
    
    print('City Distance Matrix (lower = more similar):')
    print(distance_df.round(2))
    
    #find most and least similar city pairs
    print(f'\nMost Similar Cities:')
    
    #get upper triangle of distance matrix to avoid duplicates
    upper_triangle = np.triu(distance_matrix, k=1)
    upper_triangle[upper_triangle == 0] = np.inf #replace zeros w inf
    
    min_distance_idx = np.unravel_index(np.argmin(upper_triangle), upper_triangle.shape)
    min_distance = upper_triangle[min_distance_idx]
    most_similar = (cities[min_distance_idx[0]], cities[min_distance_idx[1]])
    
    print(f'  {most_similar[0]} ↔ {most_similar[1]}: {min_distance:.3f}')
    
    print(f'\nLeast Similar Cities:')
    max_distance_idx = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
    max_distance = upper_triangle[max_distance_idx]
    least_similar = (cities[max_distance_idx[0]], cities[max_distance_idx[1]])
    
    print(f'  {least_similar[0]} ↔ {least_similar[1]}: {max_distance:.3f}')
    
    similarity_results = {
        'city_profiles': city_profiles,
        'distance_matrix': distance_df,
        'most_similar_cities': most_similar,
        'least_similar_cities': least_similar,
        'similarity_score': min_distance,
        'dissimilarity_score': max_distance
    }
    
    return similarity_results

def principal_component_analysis(df):
    """Performs PCA on city characteristics"""
    print('\nPRINCIPAL COMPONENT ANALYSIS')
    print('='*50)
    
    #prepare data for PCA
    cities = df['city'].unique()
    
    #create feature matrix
    features = []
    feature_names = ['avg_artist_popularity', 'avg_track_popularity', 'avg_followers',
                    'avg_duration', 'avg_genre_count', 'avg_track_age', 'explicit_rate']
    
    city_features = {}
    for city in cities:
        city_data = df[df['city'] == city]
        
        city_feature_vector = [
            city_data['artist_popularity'].mean(),
            city_data['track_popularity'].mean(),
            city_data['artist_followers'].mean(),
            city_data['duration_minutes'].mean(),
            city_data['genre_count'].mean(),
            city_data['track_age_years'].mean(),
            city_data['explicit'].mean()
        ]
        
        features.append(city_feature_vector)
        city_features[city] = city_feature_vector
        
    #standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    #perform pca
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)
    
    #print results
    print('PCA Results:')
    print(f'  Total variance explained: {pca.explained_variance_ratio_.sum():.3f}')
    
    for i, (variance, cumulative) in enumerate(zip(pca.explained_variance_ratio_,
                                                   np.cumsum(pca.explained_variance_ratio_))):
        print(f'  PC{i+1}: {variance:.3f} variance ({cumulative:.3f} cumulative)')
    
    #component loadings
    print(f'\nPrincipal Component Loadings:')
    components_df = pd.DataFrame(
        pca.components_[:3].T, #first 3 components
        columns=['PC1','PC2','PC3'],
        index=feature_names
    )
    
    print(components_df.round(3))
    
    #city scores on principal components
    print('\nCity Scores on Principal Components:')
    pca_scores_df = pd.DataFrame(
        pca_result[:, :3], #first 3 components
        columns=['PC1','PC2','PC3'],
        index=cities
    )
    
    print(pca_scores_df.round(3))
    
    return {
        'pca_model': pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': components_df,
        'city_scores': pca_scores_df,
        'feature_names': feature_names
    }
    
def comprehensive_statistical_analysis():
    """Main function to run comprehensive statistical analysis."""
    print("REGIONAL MUSIC DNA - STATISTICAL ANALYSIS")
    print("=" * 60)
    
    #load cleaned data
    df = load_cleaned_data()
    
    #run all analyses
    results = {}
    
    print("\n" + "="*60)
    results['descriptive_stats'] = descriptive_statistics_by_city(df)
    
    print("\n" + "="*60)
    results['genre_analysis'], results['genre_matrix'] = genre_analysis_advanced(df)
    
    print("\n" + "="*60)
    results['popularity_analysis'] = popularity_analysis(df)
    
    print("\n" + "="*60)
    results['temporal_analysis'] = temporal_analysis(df)
    
    print("\n" + "="*60)
    results['artist_tier_analysis'] = artist_tier_analysis(df)
    
    print("\n" + "="*60)
    results['correlation_analysis'] = correlation_analysis(df)
    
    print("\n" + "="*60)
    results['similarity_analysis'] = city_similarity_analysis(df)
    
    print("\n" + "="*60)
    results['pca_analysis'] = principal_component_analysis(df)
    
    #save results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../data/processed/statistical_analysis_{timestamp}.json"
    
    #convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    #only save serializable results
    serializable_results = {
        'timestamp': timestamp,
        'dataset_info': {
            'total_tracks': len(df),
            'cities': df['city'].unique().tolist(),
            'analysis_date': datetime.now().isoformat()
        },
        'genre_analysis': convert_for_json(results['genre_analysis']),
        'popularity_tests': convert_for_json(results['popularity_analysis']),
        'temporal_tests': convert_for_json(results['temporal_analysis']),
        'correlation_summary': {
            'top_correlations': convert_for_json(results['correlation_analysis'].get('top_correlations', []))
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nStatistical analysis results saved to: {results_file}")
    print(f"Statistical analysis complete!")
    
    return results, df

if __name__ == '__main__':
    analysis_results, dataset = comprehensive_statistical_analysis()