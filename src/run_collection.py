from data_collection_new import collect_all_cities_data

def collect_regional_sample():
    """Collect diverse regional sample."""
    
    # Diverse regional representation
    cities = [
        'Nashville',     # South
        'Los Angeles',   # West Coast  
        'Chicago',       # Midwest
        'Miami',         # South/Latin
        'New York',      # Northeast
        'Seattle',       # Pacific Northwest
        'Austin',        # Texas
        'Atlanta'        # Southeast
    ]
    
    print(f"🗺️ Collecting regional sample: {len(cities)} cities")
    print("=" * 60)
    
    data = collect_all_cities_data(cities, save_raw=True)
    
    print(f"\nRegional sample collection complete!")
    print(f"This gives you {len(cities)} * 200 = {len(cities) * 200} total tracks!")
    
    return data

if __name__ == "__main__":
    collect_regional_sample()