import requests, time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

url = 'http://www.imdb.com/chart/top/?ref_=nv_mv_250'
response = requests.get(url, headers=headers)
content_type = response.headers.get('Content-Type')

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Get the list of movies, strip out the extra info
movies = [movie.get_text() for movie in soup.find_all('div', class_='ipc-title')][2:-1]
movies = [movie.split('. ', 1)[1].strip() for movie in movies]

# Write the list of movies to a CSV file
df_toothpastes = pd.DataFrame(movies, columns=['Movie Title'])
df_toothpastes.to_csv('best_selling_movies.csv', index=False)

# Initialize the Chrome WebDriver
driver = webdriver.Chrome()

# Amazon URL
amazon_url = "https://www.amazon.com/"
metacritic_url = "https://www.metacritic.com/"

# List to store movie data
movie_data_list = []

# Loop through each movie title
for movie in movies:
    movie_info = {"Movie Title": movie}

    # Step 1: Scrape Amazon for prices
    driver.get(amazon_url)
    
    try:
        # Wait for the search box to be present and enter the movie title
        search_box = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "twotabsearchtextbox"))
        )
        search_box.clear()
        search_box.send_keys(movie)
        search_box.send_keys(Keys.RETURN)

        # Wait for the search results to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".s-main-slot"))
        )
        
        # Find all elements with the price class
        price_elements = driver.find_elements(By.CSS_SELECTOR, ".a-price .a-offscreen")

        # Filter out delivery charges and focus on product prices
        valid_prices = []
        for i, price_element in enumerate(price_elements):
            price_html = price_element.get_attribute('innerHTML')
            parent_html = price_element.find_element(By.XPATH, "..").get_attribute('outerHTML')
            
            # Filter logic: exclude prices with delivery-related text in their parent or adjacent containers
            if "delivery" not in parent_html.lower() and \
               "strike" not in parent_html.lower() and \
               "secondary" not in parent_html.lower() and \
               "$" in price_html:
                valid_prices.append(price_html)

        # Choose the first valid price or return "Price not found"
        price = valid_prices[0] if valid_prices else "Price not found"
        movie_info["Amazon Price"] = price

    except Exception as e:
        print(f"Could not retrieve price for {movie}: {e}")
        movie_info["Amazon Price"] = "Price not found"

    break # Remove this line to scrape all movies

# Close the driver
driver.quit()

# Create a DataFrame from the list of dictionaries
movie_df = pd.DataFrame(movie_data_list)

# Display the DataFrame
print(movie_df)

# Optionally, save the DataFrame to a CSV file
movie_df.to_csv("movie_data.csv", index=False)