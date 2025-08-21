# utils.py
import inspect
from typing import List, Optional, Tuple

import pandas as pd
from geopy.distance import geodesic

# ---------------------------
# Data models
# ---------------------------


# ---------------------------
# Loaders
# ---------------------------


def load_cities() -> pd.DataFrame:
    """
    Load U.S. cities from Kaggle's United States Cities Database.
    Normalizes columns for utils functions.
    """
    url = "https://simplemaps.com/static/data/us-cities/uscities.csv"

    df = pd.read_csv(url)
    print(df.head())

    df = df.rename(
        columns={
            "lat": "latitude",
            "lng": "longitude",
            "state_name": "region",  # match utils expectation
        }
    )

    return df[["city", "region", "latitude", "longitude"]]


df_cities = load_cities()


def load_parks() -> pd.DataFrame:
    """
    Load U.S. national parks data from local CSV file, downloading from GitHub if it doesn't exist.
    """
    import os
    
    parks_path = os.path.join(os.path.dirname(__file__), "data", "parks.csv")
    
    # Check if local file exists
    if not os.path.exists(parks_path):
        # Ensure data directory exists
        data_dir = os.path.dirname(parks_path)
        os.makedirs(data_dir, exist_ok=True)
        
        # Download from GitHub (using the same source as the original code)
        parks_url = "https://raw.githubusercontent.com/sughodke/D3-US-Graph/master/nationalparks.csv"
        print(f"Downloading parks data from {parks_url}...")
        df = pd.read_csv(parks_url)
        
        # Process the data to match expected format
        # Extract park names from details column if it exists
        if "details" in df.columns:
            df["park_name"] = df["details"].str.extract(r'"USA-National Park ([^"]+)"')
            df = df.dropna(subset=["park_name"])
            df["park_name"] = df["park_name"] + " National Park"
            
        # Create a simplified parks CSV with the columns we need
        parks_data = df[["latitude", "longitude"]].copy()
        parks_data["park_name"] = df["park_name"]
        parks_data["park_id"] = df["park_name"].str.upper().str.replace(" ", "_").str.replace("_NATIONAL_PARK", "")
        parks_data["State"] = "Unknown"  # This data source doesn't have state info
        parks_data["Acres"] = 0  # This data source doesn't have acreage info
        
        # Reorder columns to match expected format
        parks_data = parks_data[["park_id", "park_name", "State", "Acres", "latitude", "longitude"]]
        
        # Save to local cache
        parks_data.to_csv(parks_path, index=False)
        print(f"Cached parks data to {parks_path}")
        df = parks_data
    else:
        print(f"Loading parks data from local cache: {parks_path}")
        df = pd.read_csv(parks_path)
    
    # Rename columns to match expected format (in case they're not already renamed)
    if "Park Code" in df.columns:
        df = df.rename(columns={
            "Park Code": "park_id",
            "Park Name": "park_name",
            "Latitude": "latitude", 
            "Longitude": "longitude"
        })
    
    return df


def load_species() -> pd.DataFrame:
    """
    Load species data from local CSV file, downloading from GitHub if it doesn't exist.
    """
    import os
    
    species_path = os.path.join(os.path.dirname(__file__), "data", "species.csv")
    
    # Check if local file exists
    if not os.path.exists(species_path):
        # Ensure data directory exists
        data_dir = os.path.dirname(species_path)
        os.makedirs(data_dir, exist_ok=True)
        
        # Download from GitHub
        species_url = "https://raw.githubusercontent.com/brchalifour/paRkpal/master/data/species.csv"
        print(f"Downloading species data from {species_url}...")
        df = pd.read_csv(species_url, low_memory=False)
        
        # Save to local cache
        df.to_csv(species_path, index=False)
        print(f"Cached species data to {species_path}")
    else:
        print(f"Loading species data from local cache: {species_path}")
        df = pd.read_csv(species_path, low_memory=False)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "Species ID": "species_id",
        "Park Name": "park_name",
        "Park_Name": "park_name",  # Handle both possible column names
        "Scientific Name": "scientific_name",
        "Common Names": "common_name"
    })
    
    # Drop the unnamed column if it exists
    if "Unnamed: 13" in df.columns:
        df = df.drop("Unnamed: 13", axis=1)
    
    return df


def load_parks_and_biodiversity() -> pd.DataFrame:
    """
    Load combined U.S. national parks + biodiversity dataset from local CSV files.
    Each row = one species observed in one park.
    """
    
    # Load parks and species data from local CSV files
    parks_df = load_parks()
    species_df = load_species()
    
    print(f"Loaded {len(parks_df)} parks and {len(species_df)} species records")
    
    # Merge species data with park coordinates
    df = species_df.merge(parks_df[["park_name", "park_id", "State", "Acres", "latitude", "longitude"]], 
                         on="park_name", how="left")

    print(f"Combined dataset shape: {df.shape}")
    print(df.head())

    # Return columns that exist in the merged dataset
    available_columns = [
        "park_id", "park_name", "State", "Acres", "latitude", "longitude",
        "Category", "Order", "Family", "scientific_name", "common_name",
        "Nativeness", "Abundance", "Occurrence", "Seasonality"
    ]
    
    # Only include columns that actually exist in the dataframe
    columns_to_return = [col for col in available_columns if col in df.columns]
    
    return df[columns_to_return]


df_parks_bio = load_parks_and_biodiversity()


# ---------------------------
# Core Functions
# ---------------------------


def args_valid(func, args: dict) -> bool:
    """
    Check whether the given args dict satisfies all required parameters of func.
    """
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        # If a parameter has no default, it is required
        if param.default is inspect._empty and param.kind in (
            param.POSITIONAL_OR_KEYWORD,
            param.KEYWORD_ONLY,
        ):
            if name not in args:
                return False
    return True


async def get_city_location(city: str, state: str) -> Optional[Tuple[float, float]]:
    """
    Retrieves geographic coordinates for a specified U.S. city and state.

    Searches the preloaded cities dataset for an exact match (case-insensitive)
    of the provided city name and state. The dataset requires full state names
    (e.g., "Washington", "California") rather than abbreviations.

    Args:
        city: Name of the city to look up
        state: Full name of the state (e.g., "Washington", not "WA")

    Returns:
        Tuple of (latitude, longitude) as floats if city found, None otherwise
    """
    result = df_cities[
        (df_cities["city"].str.lower() == city.lower())
        & (df_cities["region"].str.lower() == state.lower())
    ]
    
    if result.empty:
        return None
    row = result.iloc[0]
    return float(row["latitude"]), float(row["longitude"])


async def get_nearest_parks_with_species(
    lat: float,
    long: float,
    species: str,
    max_results: int,
) -> List[dict]:
    """
    Finds the nearest national parks containing a specified species.

    Searches the biodiversity dataset for parks where the given species has been
    observed, calculates the distance from the provided coordinates to each park,
    and returns the closest parks sorted by distance. Includes detailed species
    information such as abundance, nativeness, and seasonality data.

    Args:
        lat: Latitude coordinate of the search origin point
        long: Longitude coordinate of the search origin point
        species: Common name of the species to search for (case-insensitive partial match)
        max_results: Maximum number of parks to return

    Sends back:
        List of dictionaries containing park information, each with:
        - park_name: Name of the national park
        - park_state: State where the park is located
        - distance: Distance from origin point in kilometers
        - species_abundance: Abundance level of the species in that park
        - species_nativeness: Whether species is native or non-native
        - species_seasonality: Seasonal presence information
    """
    matches = df_parks_bio[
        df_parks_bio["common_name"].str.contains(species, case=False, na=False)
    ].drop_duplicates(subset=["park_id"])

    if matches.empty:
        return []

    parks_with_species: List[dict] = []

    for _, row in matches.iterrows():
        park_coords = (row["latitude"], row["longitude"])
        distance_km = geodesic((lat, long), park_coords).kilometers

        parks_with_species.append(
            {
                "park_name": str(row["park_name"]),
                "park_state": str(row["State"]) if pd.notna(row["State"]) else "Unknown",
                "distance": distance_km,
                "species_abundance": str(row.get("Abundance", "Unknown")) if pd.notna(row.get("Abundance")) else "Unknown",
                "species_nativeness": str(row.get("Nativeness", "Unknown")) if pd.notna(row.get("Nativeness")) else "Unknown",
                "species_seasonality": str(row.get("Seasonality", "Unknown")) if pd.notna(row.get("Seasonality")) else "Unknown",
            }
        )

    parks_with_species.sort(key=lambda p: p["distance"])
    return parks_with_species[:max_results]


async def get_species_for_park(park_name: str, max_results: int = 20) -> List[dict]:
    """
    Retrieves a list of interesting species found in a specific national park.

    Searches the biodiversity dataset for all species observed in the given park
    and returns detailed information about each species including scientific name,
    category, abundance, and conservation status.

    Args:
        park_name: Name of the national park to search (case-insensitive partial match)
        max_results: Maximum number of species to return (default: 20)

    Returns:
        List of dictionaries containing species information, each with:
        - common_name: Common name of the species
        - category: Type of organism (Bird, Mammal, Plant, etc.)
        - abundance: Abundance level in the park
        - nativeness: Whether species is native or non-native
        - seasonality: Seasonal presence information
    """
    matches = df_parks_bio[
        df_parks_bio["park_name"].str.contains(park_name, case=False, na=False)
    ].drop_duplicates(subset=["scientific_name"])

    if matches.empty:
        return []

    species_list: List[dict] = []

    for _, row in matches.iterrows():
        species_list.append(
            {
                "common_name": str(row.get("common_name", "Unknown")) if pd.notna(row.get("common_name")) else "Unknown",
                "category": str(row.get("Category", "Unknown")) if pd.notna(row.get("Category")) else "Unknown",
                "abundance": str(row.get("Abundance", "Unknown")) if pd.notna(row.get("Abundance")) else "Unknown",
                "nativeness": str(row.get("Nativeness", "Unknown")) if pd.notna(row.get("Nativeness")) else "Unknown",
                "seasonality": str(row.get("Seasonality", "Unknown")) if pd.notna(row.get("Seasonality")) else "Unknown",
            }
        )

    # Sort by category then by common name for better organization
    species_list.sort(key=lambda s: (s["category"], s["common_name"]))
    return species_list[:max_results]


async def get_nearest_parks(
    lat: float, long: float, max_results: int = 10
) -> List[dict]:
    """
    Finds the nearest national parks to a given location.

    Calculates distances from the provided coordinates to all parks in the dataset
    and returns the closest ones sorted by distance. Useful for finding parks in
    a user's vicinity regardless of specific species interest.

    Args:
        lat: Latitude coordinate of the search origin point
        long: Longitude coordinate of the search origin point
        max_results: Maximum number of parks to return (default: 10)

    Returns:
        List of dictionaries containing park information, each with:
        - park_name: Name of the national park
        - park_state: State where the park is located
        - distance: Distance from origin point in kilometers
        - latitude: Park's latitude coordinate
        - longitude: Park's longitude coordinate
        - acres: Size of the park in acres
    """
    # Get unique parks with coordinates
    parks = df_parks_bio[
        ["park_name", "State", "latitude", "longitude", "Acres"]
    ].drop_duplicates(subset=["park_name"])

    if parks.empty:
        return []

    parks_with_distance: List[dict] = []

    for _, row in parks.iterrows():
        park_coords = (row["latitude"], row["longitude"])
        distance_km = geodesic((lat, long), park_coords).kilometers

        parks_with_distance.append(
            {
                "park_name": str(row["park_name"]),
                "park_state": str(row["State"]) if pd.notna(row["State"]) else "Unknown",
                "distance": distance_km,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "acres": int(row.get("Acres", 0)) if pd.notna(row.get("Acres")) else 0,
            }
        )

    parks_with_distance.sort(key=lambda p: p["distance"])
    return parks_with_distance[:max_results]


answer_user_tool = {
    "type": "function",
    "function": {
        "name": "answer_user",
        "description": "Answer the user's question",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Answer to the user's question",
                }
            },
            "required": ["answer"],
        },
    },
}
