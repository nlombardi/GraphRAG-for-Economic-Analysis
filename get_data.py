import pandas as pd
from typing import List
import requests


class WorldBankDataCollector:
    """
    Collects data from World Bank WDI API.

    This class handles API calls, error handling, and data normalization to create clean pandas dataframes for graph
    construction.
    """

    def __init__(self, timeout: int = 30):
        self.base_url = "https://api.worldbank.org/v2"
        self.timeout = timeout

    def get_indicators(
            self,
            countries: List[str],
            indicators: List[str],
            start_year: int = 2010,
            end_year: int = 2024
    ) -> pd.DataFrame:

        all_rows = []
        # WB accepts ISO2 or ISO3; case-insensitive, but we'll standardize to lower
        countries = [c.lower() for c in countries]

        for country in countries:
            for indicator in indicators:
                page = 1
                while True:
                    url = f"{self.base_url}/country/{country}/indicator/{indicator}"
                    params = {
                        "date": f"{start_year}:{end_year}",  # <-- FIXED
                        "format": "json",
                        "per_page": 1000,
                        "page": page,
                    }

                    try:
                        resp = requests.get(url, params=params, timeout=self.timeout)
                        resp.raise_for_status()
                        payload = resp.json()

                        # payload is [meta, data]; data may be None if no observations
                        if not isinstance(payload, list) or len(payload) < 2:
                            break

                        meta, data = payload[0], payload[1]
                        if not data:
                            break

                        for row in data:
                            all_rows.append({
                                "country_code": row["country"]["id"],
                                "country_name": row["country"]["value"],
                                "indicator_code": row["indicator"]["id"],
                                "indicator_name": row["indicator"]["value"],
                                "year": row["date"],
                                "value": row["value"],
                            })

                        # pagination
                        total_pages = int(meta.get("pages", 1))
                        if page >= total_pages:
                            break
                        page += 1

                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching {indicator} for {country}: {e}")
                        break

        return pd.DataFrame(all_rows)