"""
Create publication-ready static viz for articles and presentations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from visualization import load_analysis_data

def create_publication_figures():
    """Create high-quality figures for publication."""
    
    #set pub style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    
    df = load_analysis_data()
    
    #create main fig for publ
    fig = plt.figure(figsize=(20,16))
    
    #create grid layout
    gs = fig.add_gridspec(3,3, hspace=0.4, wspace=0.3)
    
    #calc key metrics
    genre_stats = {}
    city_metrics = {}
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        
        #genre diversity
        all_genres = []
        for genres_str in city_data['artist_genres_clean'].dropna():
            if genres_str:
                all_genres.extend([g.strip() for g in genres_str.split(',')])
        
        unique_genres = len(set(all_genres))
        genre_stats[city] = unique_genres
        
        #city metrics
        city_metrics[city] = {
            'artist_popularity': city_data['artist_popularity'].mean(),
            'track_popularity': city_data['track_popularity'].mean(),
            'track_age': city_data['track_age_years'].mean(),
            'explicit_rate': city_data['explicit'].mean() * 100
        }
        
    #fig 1 - genre diversity spectrum
    ax1 = fig.add_subplot(gs[0, :])
   
    cities_sorted = sorted(genre_stats.items(), key=lambda x: x[1], reverse=True)
    cities, diversities = zip(*cities_sorted)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cities)))
    bars = ax1.bar(cities, diversities, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax1.set_title('Regional Music DNA: Genre Diversity Across American Cities', 
                    fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel('Number of Unique Genres', fontsize=14)
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    
    #add value labels on bars
    for bar, value in zip(bars, diversities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(value), ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    #add diversity ratio line
    ax1_twin = ax1.twinx()
    ratios = [genre_stats[city] / 200 for city in cities]  #assuming 200 tracks per city
    ax1_twin.plot(cities, ratios, 'ro-', alpha=0.7, linewidth=2, markersize=6)
    ax1_twin.set_ylabel('Diversity Ratio (Genres/Track)', fontsize=14, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red', labelsize=12)
    
    #fig 2 - pop heatmap
    ax2 = fig.add_subplot(gs[1,0])
    
    popularity_data = np.array([[city_metrics[city]['artist_popularity'], 
                                city_metrics[city]['track_popularity']] for city in cities])
    
    im = ax2.imshow(popularity_data.T, cmap='RdYlBu_r', aspect='auto')
    ax2.set_xticks(range(len(cities)))
    ax2.set_xticklabels(cities, rotation=45, ha='right')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Artist Popularity', 'Track Popularity'])
    ax2.set_title('Popularity Heatmap', fontsize=14, fontweight='bold')
    
    #add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Popularity Score', fontsize=12)
    
    #fig 3 - track age dist
    ax3 = fig.add_subplot(gs[1, 1])
    
    track_ages = [city_metrics[city]['track_age'] for city in cities]
    ax3.scatter(range(len(cities)), track_ages, s=100, c=track_ages, 
                cmap='plasma', alpha=0.8, edgecolors='white', linewidth=1)
    ax3.set_xticks(range(len(cities)))
    ax3.set_xticklabels(cities, rotation=45, ha='right')
    ax3.set_ylabel('Average Track Age (Years)', fontsize=12)
    ax3.set_title('Musical Timeline', fontsize=14, fontweight='bold')
    
    #add trend line
    z = np.polyfit(range(len(cities)), track_ages, 1)
    p = np.poly1d(z)
    ax3.plot(range(len(cities)), p(range(len(cities))), "r--", alpha=0.7, linewidth=2)
    
    #fig 4 - explicit content rate
    ax4 = fig.add_subplot(gs[1, 2])
    
    explicit_rates = [city_metrics[city]['explicit_rate'] for city in cities]
    ax4.bar(range(len(cities)), explicit_rates, color='orange', alpha=0.7, 
            edgecolor='white', linewidth=1)
    ax4.set_xticks(range(len(cities)))
    ax4.set_xticklabels(cities, rotation=45, ha='right')
    ax4.set_ylabel('Explicit Content (%)', fontsize=12)
    ax4.set_title('Explicit Content Rate', fontsize=14, fontweight='bold')
    
    #fig 5 - regional comparison radar (bottom section)
    ax5 = fig.add_subplot(gs[2, :], projection='polar')
    
    #create a clean summary table/text
    ax5.axis('off')  # Remove axes for text summary
    
    #create insight boxes
    insights = [
        f"MOST DIVERSE: {cities[0]} ({diversities[0]} unique genres)",
        f"MOST FOCUSED: {cities[-1]} ({diversities[-1]} unique genres)",
        f"HIGHEST POPULARITY: {max(city_metrics.items(), key=lambda x: x[1]['artist_popularity'])[0]}",
        f"MOST CLASSIC: {max(city_metrics.items(), key=lambda x: x[1]['track_age'])[0]}",
        f"MOST MODERN: {min(city_metrics.items(), key=lambda x: x[1]['track_age'])[0]}"
    ]
    
    #add insights as text boxes
    for i, insight in enumerate(insights):
        ax5.text(0.2 * i, 0.7, insight, fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7),
                transform=ax5.transAxes, ha='center')
    
    #add title for insights
    ax5.text(0.5, 0.3, 'KEY REGIONAL MUSIC DNA INSIGHTS', 
            fontsize=20, fontweight='bold', ha='center', transform=ax5.transAxes)
    
    #add source note
    ax5.text(0.5, 0.05, 'Source: Spotify API, Last.fm, US Census | Analysis: 1,562 tracks across 8 cities', 
            fontsize=12, style='italic', alpha=0.7, ha='center', transform=ax5.transAxes)
    
    #adjust layout
    plt.tight_layout()
    
    #save with high quality
    plt.savefig('../visualizations/regional_music_dna_publication_clean.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('../visualizations/regional_music_dna_publication_clean.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.show()
    
    return fig

def create_story_figures():
    """Create individual story-focused figures for article."""
    
    df = load_analysis_data()
    
    #story fig 1 - The Great Divide (LA vs Nashville)
    fig1, ax = plt.subplots(figsize=(14,8))
    
    #compare la and nashville directly
    la_data = df[df['city'] == 'Los Angeles']
    nash_data = df[df['city'] == 'Nashville']
    
    #get genre counts
    la_genres = []
    nash_genres = []
    
    for genres_str in la_data['artist_genres_clean'].dropna():
        if genres_str:
            la_genres.etend([g.strip() for g in genres_str.split(',')])
            
    for genres_str in nash_data['artist_genres_clean'].dropna():
        if genres_str:
            nash_genres.etend([g.strip() for g in genres_str.split(',')])
            
    la_unique = len(set(la_genres))
    nash_unique = len(set(nash_genres))
    
    #create comparison
    cities_comp = ['Los Angeles\n(Innovation Hub)', 'Nashville\n(Tradition Keeper)']
    values_comp = [la_unique, nash_unique]
    colors_comp = ['#FF6B6B','#4ECDC4']
    
    bars = ax.bar(cities_comp, values_comp, color=colors_compy, alpha=0.8,
                  width=0.6, edgecolor='white', linewidth=2)
    
    #add styling
    ax.set_title('The Great Musical Divide\nGenre Diversity: Innovation vs Tradition', 
               fontsize=20, fontweight='bold', pad=30)
    ax.set_ylabel('Number of Unique Genres', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    
    #add value labels
    for bar, value in zip(bars, values_comp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value} genres', ha='center', va='bottom', 
                fontweight='bold', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
    #add ratio annotation
    ratio = la_unique / nash_unique
    ax.text(0.5, max(values_comp) * 0.8, f'LA has {ratio:.1f}x more\ngenre diversity', 
          ha='center', va='center', transform=ax.transData,
          fontsize=18, fontweight='bold',
          bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7))
   
    ax.set_ylim(0, max(values_comp) * 1.2)
    plt.tight_layout()
    plt.savefig('../visualizations/great_musical_divide.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    #story fig 2 - musical twins (miami vs austin)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
   
    miami_data = df[df['city'] == 'Miami']
    austin_data = df[df['city'] == 'Austin']
   
    #compare key metrics
    metrics = ['Artist\nPopularity', 'Track\nPopularity', 'Track Age\n(Years)', 'Explicit Rate\n(%)']
    
    miami_values = [
        miami_data['artist_popularity'].mean(),
        miami_data['track_popularity'].mean(),
        miami_data['track_age_years'].mean(),
        miami_data['explicit'].mean() * 100
    ]
    
    austin_values = [
        austin_data['artist_popularity'].mean(),
        austin_data['track_popularity'].mean(),
        austin_data['track_age_years'].mean(),
        austin_data['explicit'].mean() * 100
    ]
    
    #side-by-side comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, miami_values, width, label='Miami', 
                    color='#FF6B9D', alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x + width/2, austin_values, width, label='Austin', 
                    color='#45B7D1', alpha=0.8, edgecolor='white', linewidth=1)
    
    ax1.set_title('Musical Twins: Miami vs Austin\nSeparated by Distance, United by Sound', 
                    fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.tick_params(axis='both', labelsize=12)
    
    #add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    #similarity viz
    similarity_score = 0.85  # Based on your analysis
    ax2.pie([similarity_score, 1-similarity_score], 
            labels=[f'Musical Similarity\n{similarity_score:.1%}', f'Differences\n{1-similarity_score:.1%}'],
            colors=['#4ECDC4', '#FFE66D'], autopct='', startangle=90,
            textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax2.set_title('Cross-Country Musical Connection', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizations/musical_twins.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('Publicayion figures created successfully!')
    print('Saved: regional_music_dna_publication.png/.pdf')
    print('Saved: great_musical_divide.png')
    print('Saved: music_twins.png')

if __name__ == '__main__':
    print('Creating publication-ready visualizations...')
    
    #create main publ fig
    main_fig = create_publication_figures()
    
    #crete story-focused figs
    create_story_figures()
    
    print('\nAll publication visualizations complete!')
   