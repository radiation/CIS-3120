import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# Convert a money string like '$28,767,189' to an integer 28767189
def convert_money_to_int(money: str):
    return int(money.replace('$', '').replace(',', ''))

def get_movie_data(movie: str, driver):
    try:
        # Wait for the search box to be present and enter the movie title
        search_box = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search']"))
        )
        search_box.clear()
        search_box.send_keys(movie)
        search_box.send_keys(Keys.RETURN)

        # Wait for the search results to load and click on the first relevant result
        first_result = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/title/')]"))
        )
        first_result.click()

        # Wait for the page to load completely
        time.sleep(5)

        # Find all elements with the class 'money' & extract the text
        money_elements = driver.find_elements(By.CLASS_NAME, 'money')
        money_values = [element.text for element in money_elements]

        # There are 27 results always in a specific order
        domestic_gross = convert_money_to_int(money_values[0]) if len(money_values) > 0 else "Data not found"
        international_gross = convert_money_to_int(money_values[1]) if len(money_values) > 1 else "Data not found"
        worldwide_gross = convert_money_to_int(money_values[2]) if len(money_values) > 2 else "Data not found"
        budget = convert_money_to_int(money_values[4]) if len(money_values) > 4 else "Data not found"

        # Create & return a dictionary with the movie info
        movie_info = {
            "Movie Title": movie,
            "Budget": budget,
            "Domestic Gross": domestic_gross,
            "International Gross": international_gross,
            "Worldwide Gross": worldwide_gross
        }
        return movie_info

    except Exception as e:
        print(f"Could not retrieve data for {movie} from Box Office Mojo: {e}")
        return None

def main():

    # Headers for making requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }

    # Retrieve and parse the IMDb Top 250 page
    url = 'http://www.imdb.com/chart/top/?ref_=nv_mv_250'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get the list of movies & remove the extra entries & extra characters
    movies = [movie.get_text() for movie in soup.find_all('div', class_='ipc-title')][2:-1]
    movies = [movie.split('. ', 1)[1].strip() for movie in movies]

    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome()

    # Box Office Mojo URL
    boxofficemojo_url = "https://www.boxofficemojo.com/"

    # List to store movie data
    movie_data_list = []

    # Loop through each movie title
    for movie in movies:
        driver.get(boxofficemojo_url)
        movie_info = get_movie_data(movie, driver)
        if movie_info:
            movie_data_list.append(movie_info)

    # Close the driver
    driver.quit()

    # Create a DataFrame from the list of dictionaries
    movie_df = pd.DataFrame(movie_data_list)

    # Display the DataFrame
    print(movie_df.describe())

    # Optionally, save the DataFrame to a CSV file
    movie_df.to_csv("movie_data_boxofficemojo.csv", index=False)

if __name__ == "__main__":
    main()