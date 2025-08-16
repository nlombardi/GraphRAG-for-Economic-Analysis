import pandas as pd
from typing import List
import requests


class WorldBankDataCollector:
    """
    Collects data from World Bank WDI API.

    This class handles API calls, error handling, and data normalization to create clean pandas dataframes for graph
    construction.
    """

    def __init__(self):
        self.base_url = "https://api.worldbank.org/v2"

    def get_indicators(self, countries: List[str], indicators: List[str],
                       start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """
        Get the WDI data for specified countries and indicators.

        :param countries: List of countries to collect data for.
        :param indicators: List of indicators to collect data for.
        :param start_year: Beginning year to collect data for.
        :param end_year: Ending year to collect data for.
        :return: Pandas DataFrame with columns: country_code, country_name, indicator_code, indicator_name, year, value
        """
        all_data = []

        # Iterate through countries and indicators to construct API urls
        for country in countries:
            for indicator in indicators:
                url = f"{self.base_url}/country/{country}/indicator/{indicator}"

                params = {
                    'data': f"{start_year}:{end_year}",
                    'format': 'json',
                    'per_page': 1000,
                }

                try:
                    response = requests.get(url, params=params)
                    data = response.json()

                    # WB WDI API returns [metadata, data]
                    if len(data) > 1 and data[1]:
                        for row in data[1]:
                            # Extract relevant data
                            all_data.append({
                                'country_code': row['country']['id'],
                                'country_name': row['country']['value'],
                                'indicator_code': row['indicator']['id'],
                                'indicator_name': row['indicator']['value'],
                                'year': row['date'],
                                'value': row['value']
                            })
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching {indicator} for {country}: {e}")

        return pd.DataFrame(all_data)