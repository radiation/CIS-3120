import os

import pandas as pd
import requests

API_KEY = os.getenv('WEATHER_API_KEY')

# Take city as a string and return a weather row as a dictionary
def get_weather_for_city(city: str) -> dict:
    url = f"https://api.weatherapi.com/v1/current.json?q={city}&key={API_KEY}"
    try:
        current_weather: dict = requests.get(url).json()['current']
        return {
            'city': city,
            'temp_f': current_weather['temp_f'],
            'feels_like': current_weather['feelslike_f'], # Can't calc in DF but still nice to see
            'condition': current_weather['condition']['text'],
            'wind_mph': current_weather['wind_mph'],
            'humidity': current_weather['humidity'],
            'uv': current_weather['uv']
        }
    except KeyError:
        print(f"Error fetching weather data for {city}.")
        return None

def main():
    cities: list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    # If you happen to put in Xi'an, China or something, it'll break so we need to filter out the None values
    weather_data: list = [city_data for city in cities if (city_data := get_weather_for_city(city)) is not None]

    # Print DF & write to CSV
    df = pd.DataFrame(weather_data)
    print(df.describe().round(2))
    df.to_csv('city_weather.csv', index=False)

if __name__ == "__main__":
    main()