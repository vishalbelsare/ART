# utils.py
import inspect
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from geopy.distance import geodesic

# ---------------------------
# Data models
# ---------------------------


@dataclass
class ParkSpecies:
    park_name: str
    park_state: str
    distance: float  # kilometers
    species_abundance: str
    species_nativeness: str
    species_seasonality: str


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


def load_parks_and_biodiversity() -> pd.DataFrame:
    """
    Load combined U.S. national parks + biodiversity dataset.
    Each row = one species observed in one park.
    Source: GitHub educational repositories with National Parks biodiversity data
    """
    import os

    local_path = "./datasets/np_species.csv"

    # Check if dataset is already saved locally
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"Loaded dataset from {local_path}")
    else:
        # Ensure local directory exists
        os.makedirs("./datasets", exist_ok=True)

        # Load individual datasets and combine them
        print("Downloading and combining biodiversity datasets...")

        # Load species observations (park_name + scientific_name + observations)
        observations_url = "https://raw.githubusercontent.com/Kate-Pol/Biodiversity-in-National-Parks/master/observations.csv"
        observations = pd.read_csv(observations_url)

        # Load species info (scientific_name + category + common_names + conservation_status)
        species_url = "https://raw.githubusercontent.com/Kate-Pol/Biodiversity-in-National-Parks/master/species_info.csv"
        species_info = pd.read_csv(species_url)

        # Load park coordinates (park names + lat/lon)
        parks_url = "https://raw.githubusercontent.com/sughodke/D3-US-Graph/master/nationalparks.csv"
        parks_coords = pd.read_csv(parks_url)

        # Extract park names from details column
        parks_coords["park_short_name"] = parks_coords["details"].str.extract(
            r'"USA-National Park ([^"]+)"'
        )
        parks_coords = parks_coords[
            ["latitude", "longitude", "park_short_name"]
        ].dropna()

        # Create a mapping for park name matching
        name_mapping = {
            "Great Smoky Mountains National Park": "Great Smoky Mountains",
            "Yosemite National Park": "Yosemite",
            "Bryce National Park": "Bryce Canyon",
            "Yellowstone National Park": "Yellowstone",
        }

        # Merge observations with species info
        df = observations.merge(species_info, on="scientific_name", how="left")

        # Add short park names for matching with coordinates
        df["park_short_name"] = df["park_name"].map(name_mapping)

        # Merge with park coordinates using short names
        df = df.merge(parks_coords, on="park_short_name", how="left")

        # Drop the temporary column
        df = df.drop("park_short_name", axis=1)

        # Add missing columns to match expected format
        df = df.rename(
            columns={
                "park_name": "park_name",
                "scientific_name": "scientific_name",
                "common_names": "common_name",
                "category": "Category",
                "observations": "Abundance",
                "conservation_status": "Nativeness",
            }
        )

        # Add placeholder columns for missing data
        df["park_id"] = df["park_name"].str.upper().str.replace(" ", "_")
        df["State"] = "Unknown"
        df["Acres"] = 0
        df["Order"] = "Unknown"
        df["Family"] = "Unknown"
        df["Occurrence"] = "Present"
        df["Seasonality"] = "Unknown"

        # Save to local path
        df.to_csv(local_path, index=False)
        print(f"Downloaded and saved combined dataset to {local_path}")

    print(f"Dataset shape: {df.shape}")
    print(df.head())

    return df[
        [
            "park_id",
            "park_name",
            "State",
            "Acres",
            "latitude",
            "longitude",
            "Category",
            "Order",
            "Family",
            "scientific_name",
            "common_name",
            "Nativeness",
            "Abundance",
            "Occurrence",
            "Seasonality",
        ]
    ]


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
    Returns [lat, lon] for a given city and state.
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
) -> List[ParkSpecies]:
    """
    Returns nearest parks where a species is present.
    Filters df_parks_bio for species and computes distances.
    """
    matches = df_parks_bio[
        df_parks_bio["common_name"].str.contains(species, case=False, na=False)
    ].drop_duplicates(subset=["park_id"])

    if matches.empty:
        return []

    parks_with_species: List[ParkSpecies] = []

    for _, row in matches.iterrows():
        park_coords = (row["latitude"], row["longitude"])
        distance_km = geodesic((lat, long), park_coords).kilometers

        parks_with_species.append(
            ParkSpecies(
                park_name=row["park_name"],
                park_state=row["State"],
                distance=distance_km,
                species_abundance=row.get("Abundance", "Unknown"),
                species_nativeness=row.get("Nativeness", "Unknown"),
                species_seasonality=row.get("Seasonality", "Unknown"),
            )
        )

    parks_with_species.sort(key=lambda p: p.distance)
    return parks_with_species[:max_results]


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
