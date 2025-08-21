# utils.py
import pandas as pd
from dataclasses import dataclass
from geopy.distance import geodesic
from typing import Optional, List, Tuple
from datasets import load_dataset


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


def load_cities(path: str = "uscities.csv") -> pd.DataFrame:
    """
    Load U.S. cities from Kaggle's United States Cities Database.
    Normalizes columns for utils functions.
    """
    df = pd.read_csv(path)

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
    Source: Solshine/Biodiversity_In_National_Parks
    """
    ds = load_dataset("Solshine/Biodiversity_In_National_Parks", split="train")
    df = ds.to_pandas()

    df = df.rename(
        columns={
            "Park Code": "park_id",
            "Park Name": "park_name",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Scientific Name": "scientific_name",
            "Common Names": "common_name",
        }
    )

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
    max_results: int = 5,
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
