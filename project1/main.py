import requests
import pandas as pd
import csv
from bs4 import BeautifulSoup
import numpy as np


# Function to print the top 5
def print_top_5(df: pd.DataFrame, sport: str, gender: str, tallest: bool):
    filtered_df = df[(df['Sport'] == sport) & (df['Gender'] == gender)]


# Main function
def main():

    # Initialize numpy arrays to store player data
    genders = np.array([])
    sports = np.array([])
    schools = np.array([])
    names = np.array([])
    heights = np.array([])

    # Read programs from the csv file
    with open('programs.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Visual progress indicator
            print(f'Processing {row[2]}: {row[0]} {row[1]}')

            # Get player data
            response = requests.get(row[3], headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')

            # We use tmp variables to keep track of each player list and validate that they're the same length
            tmp_names = [name.get_text(strip=True) for name in soup.find_all('td', class_='sidearm-table-player-name')]
            tmp_heights = [height['data-sort'] for height in soup.find_all('td', class_='height')]
            if (len(tmp_names) != len(tmp_heights)):
                print(f'Error: {row[2]} {row[1]} {row[0]}')

            # Update the main lists
            names += tmp_names
            heights += tmp_heights
            genders += [row[0]] * len(tmp_names)
            sports += [row[1]] * len(tmp_names)
            schools += [row[2]] * len(tmp_names)

    # Print all list lengths so we know they're equal
    print(f'{len(names)} names\n{len(heights)} heights\n{len(sports)} sports\n{len(schools)} schools\n{len(genders)} genders')

    # Create a pandas dataframe
    player_data = pd.DataFrame(
        {
            'Gender': genders,
            'Sport': sports,
            'School': schools,
            'Name': names,
            'Height': heights
        }
    )

    player_data.to_csv('players.csv', index=False)

    # Filter the DataFrame to include only male swimmers
    male_swimmers = player_data[(player_data['Gender'] == 'M') & (player_data['Sport'] == 'Swimming')]

    # Sort the filtered DataFrame by Height in descending order
    male_swimmers_sorted = male_swimmers.sort_values(by='Height', ascending=False)

    # Select the top five tallest male swimmers
    print(male_swimmers_sorted.head(5))

    # Ensure the Height column is numeric and filter out any zero values
    player_data['Height'] = pd.to_numeric(player_data['Height'])
    player_data_filtered = player_data[player_data['Height'] != 0]

    # Group by Gender and Sport, then calculate the average height
    average_heights = player_data_filtered.groupby(['Gender', 'Sport'])['Height'].mean().reset_index()

    # Rename the Height column to Average Height for clarity
    average_heights.rename(columns={'Height': 'Average Height'}, inplace=True)

    # Print the results
    print(average_heights)

if __name__ == '__main__':
    main()