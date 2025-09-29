"""
Advanced visualization module for Regional Music DNA project.
Create compelling, publication-ready visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import os
import warnings
warnings.filterwarnings('ignore')

#set style
plt.style.use('default')
sns.set_palette('husl')

def load_analysis_data():
    """Load cleaned data and analysis results."""
    #load cleaned data
    processed_dir = '../data/processed/'
    cleaned_files = [f for f in os.listdir(processed_dir) if f.startswith('spotify_cleaned_')]
    latest_cleaned = sorted(cleaned_files)[-1]
    df = pd.read_csv(os.path.join(processed_dir, latest_cleaned))
    
    print(f"📊 Loaded: {len(df):,} tracks from {df['city'].nunique()} cities")
    return df

def create_musical_dna_fingerprint(df):
    """Create radar charts showin each city's musical DNA fingerprint"""
    print('\nCREATING MUSICAL DNA FINGERPRINTS')
    print('='*50)
    
    #calculate city profiles
    cities = df['city'].unique()
    metrics = ['Popularity','Diversity','Modernity','Mainstream Appeal']
    
    city_profiles = {}
    
    for city in cities:
        city_data = df[df['city'] == city]
        
        #calc normalized metrics (0-100) scale
        all_genres = []
        for genres_str in city_data['artist_genres_clean'].dropna():
            if genres_str:
                all_genres.extend([g.strip() for g in genres_str.split(',')])
        unique_genres = len(set(all_genres))
        
        profile = {
            'Popularity': city_data['track_popularity'].mean(),
            'Diversity': min((unique_genres/109)*100,100),
            'Modernity': max(0, 100 - ((city_data['track_age_years'].mean()/30) * 100)),
            'Appeal': city_data['artist_popularity'].mean()
        }
        
        city_profiles[city] = profile
        
    #create radar chart using plotly
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=cities,
        specs=[[{'type':'polar'}]*4]*2,
        vertical_spacing=0.25,
        horizontal_spacing=0.15
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    for i, city in enumerate(cities):
        row = (i // 4)+1
        col = (i%4)+1
        
        profile = city_profiles[city]
        values = list(profile.values())
        values.append(values[0])
        
        labels = list(profile.keys())
        labels.append(labels[0])
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=city,
                line=dict(color=colors[i % len(colors)], width=4),
                fillcolor=colors[i % len(colors)],
                opacity=0.4
            ),
            row=row, col=col
        )
        
    fig.update_layout(
        title={
            'text': "Regional Music DNA Fingerprints<br>Each City's Unique Musical Characteristics",
            'x':0.5,
            'font': {'size':24}
        },
        height=800,
        width=1200,
        showlegend=False,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    #for better readability
    for i in range(len(cities)):
        polar_name = f'polar{i+1 if i > 0 else ""}'
        fig.update_layout({
            polar_name: dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 100],
                    tickmode='linear',
                    tick0=0,
                    dtick=50,
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    rotation=45
                )
            )
        })
    
    return fig, city_profiles

def create_genre_diversity_spectrum(df):
    """create an elegant genre diversity viz"""
    print('\nCREATING GENRE DIVERSITY SPECTRUM')
    print('='*50)
    
    #calc genre diversity metrics
    genre_stats = {}
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        
        all_genres = []
        for genres_str in city_data['artist_genres_clean'].dropna():
            if genres_str:
                all_genres.extend([g.strip() for g in genres_str.split(',')])
                
        unique_genres = len(set(all_genres))
        total_tracks = len(city_data)
        
        genre_stats[city] = {
            'unique_genres': unique_genres,
            'diversity_ratio': unique_genres/total_tracks,
            'total_tracks': total_tracks
        }
    
    #create interactive viz
    cities = list(genre_stats.keys())
    unique_genres = [genre_stats[city]['unique_genres'] for city in cities]
    diversity_ratios = [genre_stats[city]['diversity_ratio'] for city in cities]
    
    #sort by diversity
    sorted_data = sorted(zip(cities, unique_genres, diversity_ratios),
                         key=lambda x: x[1], reverse=True)
    cities_sorted, genres_sorted, ratios_sorted = zip(*sorted_data)
    
    #create bubble chart
    fig = go.Figure()
    
    y_positions = []
    x_positions = []
    
    for i, (city, genres, ratio) in enumerate(zip(cities_sorted, genres_sorted, ratios_sorted)):
        x_pos = i * 1.5
        y_pos = genres + (i % 3 - 1)*8
        
        x_positions.append(x_pos)
        y_positions.append(y_pos)
     
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=y_positions,
        mode='markers+text',
        marker=dict(
            size=[r*400 for r in ratios_sorted], #size based on ratio
            color=genres_sorted,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Unique Genres'),
            opacity=0.7,
            line=dict(width=3, color='white')
        ),
        text=cities_sorted,
        textposition='middle center',
        textfont=dict(size=12, color='black', family='Arial Black'),
        hovertemplate="<b>%{text}</b><br>" +
                     "Unique Genres: %{y:.0f}<br>" +
                     "Diversity Ratio: %{customdata:.3f}<br>" +
                     "Tracks: " + str([genre_stats[city]['total_tracks'] for city in cities_sorted]) + "<br>" +
                     "<extra></extra>",
        customdata=ratios_sorted
    ))
    
    fig.update_layout(
        title={
            'text': "Regional Music Diversity Spectrum<br><sub>Genre Variety Across American Cities</sub>",
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Cities (Ranked by Diversity)",
            tickvals=x_positions,
            ticktext=cities_sorted,
            tickangle=45,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Number of Unique Genres",
            tickfont=dict(size=12)
        ),
        height=700,
        width=1000,
        template="plotly_white",
        margin=dict(l=80, r=80, t=100, b=120)
    )
    
    return fig, genre_stats
    
def create_city_similarity_network(df):
    """create a network viz showing city musical similarities"""
    print('\n CREATING CITY SIMILARITY NETWORK')
    print('='*50)
    
    #calc city profiles
    cities = df['city'].unique()
    city_features = []
    
    for city in cities:
        city_data = df[df['city'] == city]
        features = [
            city_data['artist_popularity'].mean(),
            city_data['track_popularity'].mean(),
            city_data['artist_followers'].mean(),
            city_data['duration_minutes'].mean(),
            city_data['genre_count'].mean(),
            city_data['track_age_years'].mean(),
            city_data['explicit'].mean()
        ]
        city_features.append(features)
        
    #standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(city_features)
    
    #calc similarity matrix (using correlation for similarity)
    similarity_matrix = np.corrcoef(features_scaled)
    
    #create network graph
    G = nx.Graph()
    
    #add nodes
    for city in cities:
        G.add_node(city)
        
    #add edges for strong similarities (correlation > 0.3)
    threshold = 0.3
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            similarity = similarity_matrix[i,j]
            if similarity > threshold:
                G.add_edge(cities[i], cities[j], weight=similarity)
                
    #calc layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    #extract edge info
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G[edge[0]][edge[1]]['weight'])
        
    #create edge traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    #create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x,y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        #size based on number of connections
        node_size.append(len(list(G.neighbors(node)))*10+20)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_size,
            color='lightblue',
            line=dict(width=2, color='white')
        )
    )
    
    #create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title="Musical City Similarity Network<br>Connected cities share similar musical characteristics",
                       font_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Node size = number of similar cities<br>Connections = correlation > 0.3",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="gray", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig, similarity_matrix
    
def create_temporal_evolution_map(df):
    """create a viz showing musical evolution over time by city."""
    print('='*50)
    
    #prepare data for timeline viz
    timeline_data = []
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        
        #group by decade
        decades = city_data.groupby('release_decade').agg({
            'track_id':'count',
            'artist_popularity': 'mean',
            'track_popularity': 'mean',
            'genre_count': 'mean'
        }).reset_index()
        
        for _, decade_data in decades.iterrows():
            timeline_data.append({
                'city': city,
                'decade': decade_data['release_decade'],
                'track_count': decade_data['track_id'],
                'avg_artist_popularity': decade_data['artist_popularity'],
                'avg_track_popularity': decade_data['track_popularity'],
                'avg_genre_diversity': decade_data['genre_count']
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    #create animated scatter plot
    fig = px.scatter(
        timeline_df,
        x='avg_artist_popularity',
        y='avg_track_popularity',
        size='track_count',
        color='city',
        animation_frame='decade',
        hover_name='city',
        hover_data=['track_count','avg_genre_diversity'],
        title='Music Evolution Through Time<br>Artist vs Track Popularity by Decade',
        labels={
            'avg_artist_popularity': 'Average Artist Popularity',
            'avg_track_popularity': 'Average Track Popularity',
            'track_count': 'Number of Tracks'
        }
    )
    
    fig.update_layout(height=600)
    
    return fig, timeline_df
    
def create_popularity_vs_diversity_matrix(df):
    """create a matrix showing the relationship between popularity and diversity."""
    print('\nCREATING POPULARITY VS DIVERSITY MATRIX')
    print('='*50)
    
    #calc metrics for each city
    city_metrics = {}
    
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        
        #calc genre diversiry
        all_genres = []
        for genres_str in city_data['artist_genres_clean'].dropna():
            if genres_str:
                all_genres.extend([g.strip() for g in genres_str.split(',')])
                
        unique_genres = len(set(all_genres))
        
        city_metrics[city] = {
            'avg_artist_popularity': city_data['artist_popularity'].mean(),
            'avg_track_popularity': city_data['track_popularity'].mean(),
            'genre_diversity': unique_genres,
            'avg_followers': city_data['artist_followers'].mean(),
            'avg_track_age': city_data['track_age_years'].mean()
        }
        
    #convert to df
    metrics_df = pd.DataFrame(city_metrics).T
    
    #create scatterplot matrix
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Artist Popularity vs Genre Diversity',
                        'Track Popularity vs Genre Diversity',
                       'Artist Followers vs Genre Diversity',
                       'Track Age vs Genre Diversity'
        ],
        specs=[[{'type':'scatter'}, {'type':'scatter'}],
               [{'type':'scatter'}, {'type':'scatter'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    cities = list(city_metrics.keys())
    colors = px.colors.qualitative.Set3
    
    #plot1 - artist pop vs genre diversity
    fig.add_trace(
        go.Scatter(
            x=metrics_df['avg_artist_popularity'],
            y=metrics_df['genre_diversity'],
            mode='markers+text',
            text=cities,
            textposition='top center',
            textfont=dict(size=11, color='black'),
            marker=dict(size=12, color=colors[:len(cities)],
                        line=dict(width=2, color='white')),
            name='Artist Popularity',
            showlegend=False
        ),
        row=1, col=1
    )
    
    #plot2 - track pop vs genre diversity
    fig.add_trace(
        go.Scatter(
            x=metrics_df['avg_track_popularity'],
            y=metrics_df['genre_diversity'],
            mode='markers+text',
            text=cities,
            textposition="top center",
            textfont=dict(size=11, color='black'),
            marker=dict(size=15, color=colors[:len(cities)],
                       line=dict(width=2, color='white')),
            name='Track Popularity',
            showlegend=False
        ),
        row=1, col=2
    )
    
    #plot 3 - artist followers vs genre diversity (log scale)
    fig.add_trace(
        go.Scatter(
            x=np.log10(metrics_df['avg_followers']),
            y=metrics_df['genre_diversity'],
            mode='markers+text',
            text=cities,
            textposition="top center",
            textfont=dict(size=11, color='black'),
            marker=dict(size=15, color=colors[:len(cities)],
                       line=dict(width=2, color='white')),
            name='Artist Followers',
            showlegend=False
        ),
        row=2, col=1
    )
    
    #plot 4 - track age vs genre diversity
    fig.add_trace(
        go.Scatter(
            x=metrics_df['avg_track_age'],
            y=metrics_df['genre_diversity'],
            mode='markers+text',
            text=cities,
            textposition="top center",
            textfont=dict(size=11, color='black'),
            marker=dict(size=15, color=colors[:len(cities)],
                       line=dict(width=2, color='white')),
            name='Track Age',
            showlegend=False
        ),
        row=2, col=2
    )
    
    #update layout
    fig.update_xaxes(title_text="Artist Popularity", row=1, col=1, title_font=dict(size=14))
    fig.update_xaxes(title_text="Track Popularity", row=1, col=2, title_font=dict(size=14))
    fig.update_xaxes(title_text="Log(Artist Followers)", row=2, col=1, title_font=dict(size=14))
    fig.update_xaxes(title_text="Average Track Age (Years)", row=2, col=2, title_font=dict(size=14))
    
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_yaxes(title_text="Genre Diversity", row=row, col=col, title_font=dict(size=14))
    
    fig.update_layout(
        title={
            'text': "Musical Characteristics Matrix<br><sub>Exploring Relationships Between Popularity and Diversity</sub>",
            'x': 0.5,
            'font': {'size': 18}
        },
        height=1000,  # Much taller
        width=1200,   # Much wider
        font=dict(size=12),
        margin=dict(l=80, r=80, t=120, b=80)  # Much larger margins
    )
    
    return fig, metrics_df
    
def create_comprehensive_dashboard():
    """create a comprehensive dashboard w all vizs"""
    print('\nCREATING COMPREHENSIVE VISUALIZATION DASHBOARD')
    print('='*60)
    
    #load data
    df = load_analysis_data()
    
    #create all viz
    print('Creating visualizations...')
    
    #1. musical dna fingerprints
    dna_fig, city_profiles = create_musical_dna_fingerprint(df)
    
    #2. genre diversity spectrum
    diversity_fig, genre_stats = create_genre_diversity_spectrum(df)
    
    #3. city similarity network
    network_fig, similarity_matrix = create_city_similarity_network(df)
    
    #4. temporal evolution
    temporal_fig, timeline_df = create_temporal_evolution_map(df)
    
    #5. pop vs diversity matrix
    matrix_fig, matrics_df = create_popularity_vs_diversity_matrix(df)
    
    #save all viz
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = f'../visualizatioms/regional_music_dna_{timestamp}/'
    
    os.makedirs(viz_dir, exist_ok=True)
    
    #save as html files
    dna_fig.write_html(f"{viz_dir}musical_dna_fingerprints.html")
    diversity_fig.write_html(f"{viz_dir}genre_diversity_spectrum.html")
    network_fig.write_html(f"{viz_dir}city_similarity_network.html")
    temporal_fig.write_html(f"{viz_dir}temporal_evolution.html")
    matrix_fig.write_html(f"{viz_dir}popularity_diversity_matrix.html")
    
    print(f'\nVisualizations saved to: {viz_dir}')
    
    #create summary report
    summary = {
        'timestamp': timestamp,
        'total_tracks': len(df),
        'cities_analyzed': df['city'].nunique(),
        'visualizations_created': 5,
        'city_profiles': city_profiles,
        'genre_statistics': genre_stats,
        'files_created': [
            'musical_dna_fingerprints.html',
            'genre_diversity_spectrum.html',
            'city_similarity_network.html',
            'temporal_evolution.html',
            'popularity_diversity_matrix.html'
        ]
    }
    
    return summary, {
        'dna_fingerprints': dna_fig,
        'diversity_spectrum': diversity_fig,
        'similarity_network': network_fig,
        'temporal_evolution': temporal_fig,
        'popularity_matrix': matrix_fig
    }
    
if __name__ == '__main__':
    summary, visualizations = create_comprehensive_dashboard()
    
    print(f'\nVISUALIZATION DASHBOARD COMPLETE!')
    print(f'Created {summary["visualizations_created"]} interactive visualizations')
    print(f'Analyzed {summary["cities_analyzed"]} cities')
    print(f'Processed {summary["total_tracks"]:,} tracks')
    print(f'\nFiles created:')
    for file in summary['files_created']:
        print(f'  {file}')
    
    print(f'\nReady for publication analysis!')