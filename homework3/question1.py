import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

API_KEY = os.getenv('WEATHER_API_KEY')

# Return a pop row as a dictionary
def parse_pop_row(row) -> dict:
    cells = row.find_all('td')
    return {
        'city': row['id'],
        'country': cells[0].get_text(strip=True), 
        'census': convert_to_int(cells[1].get_text(strip=True)), 
        'city_proper': convert_to_int(cells[6].get_text(strip=True)), 
        'metro_area': convert_to_int(cells[9].get_text(strip=True))  
    }

# Return a weather row as a dictionary
def parse_weather_row(city) -> dict:
    url = f"https://api.weatherapi.com/v1/current.json?q={city}&key={API_KEY}"
    try:
        current_weather = requests.get(url).json()['current']
        return {
            'city': city,
            'temp_f': current_weather['temp_f'],
            'condition': current_weather['condition']['text'],
            'wind_mph': current_weather['wind_mph'],
            'humidity': current_weather['humidity']
        }
    except KeyError:
        print(f"Error fetching weather data for {city}.")
        return None

def convert_to_int(value: str) -> int:
    num = value.replace(',', '')
    try:
        return int(num)
    except ValueError:
        return 0

def main():
    # For html scraping; not needed for json calls
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }
    url = "https://en.wikipedia.org/wiki/List_of_largest_cities"
    print(f"Fetching {url}...")

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # This is a little convoluted, but it's objectively more efficient than using a for loop
    city_pop_data = [parse_pop_row(row) for row in soup.find_all('tr') if row.has_attr('id')]

    # This one took a bit of magic to filter out the None values without calling the function twice
    city_weather_data = [result for city in city_pop_data if (result := parse_weather_row(city['city'])) is not None]

    # Create dataframes and merge them
    df_population = pd.DataFrame(city_pop_data)
    df_weather = pd.DataFrame(city_weather_data)
    df_combined = pd.merge(df_population, df_weather, on='city')

    print(df_combined)

    df_combined.to_csv('city_data.csv', index=False)

if __name__ == "__main__":
    main()