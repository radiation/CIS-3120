import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the Chrome WebDriver
driver = webdriver.Chrome()

# List to store movie data
movie_data_list = []

# Movie title
movie = "The Shawshank Redemption"

# Navigate to Box Office Mojo
driver.get("https://www.boxofficemojo.com/")

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

    # Find all elements with the class 'money'
    money_elements = driver.find_elements(By.CLASS_NAME, 'money')

    # Extract the text from these elements
    money_values = [element.text for element in money_elements]

    # Assuming the positions are consistent:
    # - Budget is often first or second
    # - Domestic gross might be in a specific position, say 0 or 1
    # - International gross might follow domestic

    [print(f"Money value {i}: {value}") for i, value in enumerate(money_values)]

    # Example (you might need to adjust indices based on actual structure):
    domestic_gross = money_values[0] if len(money_values) > 0 else "Data not found"
    international_gross = money_values[1] if len(money_values) > 1 else "Data not found"
    worldwide_gross = money_values[2] if len(money_values) > 2 else "Data not found"
    budget = money_values[4] if len(money_values) > 4 else "Data not found"

    # Print the extracted values
    print(f"Budget: {budget}")
    print(f"Domestic Gross: {domestic_gross}")
    print(f"International Gross: {international_gross}")
    print(f"Worldwide Gross: {worldwide_gross}")

    # Create a dictionary with the movie info
    movie_info = {
        "Movie Title": movie,
        "Budget": budget,
        "Domestic Gross": domestic_gross,
        "International Gross": international_gross,
        "Worldwide Gross": worldwide_gross
    }

    # Append the movie info to the list
    movie_data_list.append(movie_info)

except Exception as e:
    print(f"Could not retrieve data for {movie} from Box Office Mojo: {e}")

# Close the driver
driver.quit()

# Create a DataFrame from the list of dictionaries
movie_df = pd.DataFrame(movie_data_list)

# Display the DataFrame
print(movie_df)

# Optionally, save the DataFrame to a CSV file
movie_df.to_csv("movie_data.csv", index=False)
