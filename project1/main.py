import csv
from typing import List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Define the combinations of Gender and Sport
combinations = [
    ('M', 'Volleyball'),
    ('W', 'Volleyball'),
    ('M', 'Swimming'),
    ('W', 'Swimming')
]

# Handle each combination; we also need to pass the df so we can filter it
def handle_combination(gender: str, sport: str, df: pd.DataFrame):
    print(f"\nProcessing {gender} {sport}...")
    filtered_df: pd.DataFrame = df[(df['Gender'] == gender) & (df['Sport'] == sport)]
    filename: str = f"{gender}_{sport}.csv"
    filtered_df.to_csv(filename, index=False)
    print(f"Saved {filename}")
    
    # Calculate and print the average height
    if not filtered_df.empty:  # Check if the filtered DataFrame is not empty
        average_height: pd.DataFrame = filtered_df['Height'].mean()
        print(f"Average height for {gender} {sport}: {average_height:.2f}")

        # Print the five tallest
        tallest = filtered_df.nlargest(5, 'Height')
        print(f"Five tallest in {gender} {sport}:")
        print(tallest[['Name', 'Height']])

        # Print the five shortest
        shortest = filtered_df.nsmallest(5, 'Height')
        print(f"Five shortest in {gender} {sport}:")
        print(shortest[['Name', 'Height']])
    else:
        print(f"No data for {gender} {sport}")

# Main function
def main():
    # Initialize an empty base numpy array
    base_array: np.ndarray = np.empty((0, 5), dtype=object)

    # Read programs from the csv file
    with open('programs.csv', mode='r') as file:
        csv_reader: csv.reader = csv.reader(file)

        for row in csv_reader:
            # Visual progress indicator because this takes a bit 
            print(f'Processing {row[2]}: {row[0]} {row[1]}')

            # Get player data
            response: requests.Response = requests.get(row[3], headers={'User-Agent': 'Mozilla/5.0'})
            soup: BeautifulSoup = BeautifulSoup(response.content, 'html.parser')

            # Temporary lists to hold player data; replace double spaces in names with single spaces because the inconsistency annoys me
            tmp_names: List[str] = [name.get_text(strip=True).replace("  ", " ") for name in soup.find_all('td', class_='sidearm-table-player-name')]
            tmp_heights: List[int] = [int(height['data-sort']) for height in soup.find_all('td', class_='height')]
            
            if len(tmp_names) != len(tmp_heights):
                print(f'Error: {row[2]} {row[1]} {row[0]} - Number of names and heights do not match')
                continue  # Skip this iteration if there's a mismatch; not sure if there's a better way to handle this

            # Convert temporary lists to numpy arrays so we can concatenate them efficiently
            tmp_array: np.ndarray = np.array([[row[0], row[1], row[2], name, height] for name, height in zip(tmp_names, tmp_heights)], dtype=object)

            # Check if tmp_array is not empty before concatenating; vstack just concatenates vertically
            if tmp_array.size > 0:
                base_array = np.vstack((base_array, tmp_array))

    # Convert the base numpy array to a DataFrame
    df: pd.DataFrame = pd.DataFrame(base_array, columns=['Gender', 'Sport', 'School', 'Name', 'Height'])

    # Calculate the average height for each sport/gender combination, excluding zeros
    grouped_averages: pd.DataFrame = df[df['Height'] > 0].groupby(['Gender', 'Sport'])['Height'].mean().reset_index()
    grouped_averages = grouped_averages.rename(columns={'Height': 'Average_Height'})

    # Merge the average heights back into the original DataFrame
    df = df.merge(grouped_averages, on=['Gender', 'Sport'], how='left')

    # Replace zero heights with the corresponding average height
    df['Height'] = df.apply(lambda row: row['Average_Height'] if row['Height'] == 0 else row['Height'], axis=1)

    # Drop the Average_Height column as it's no longer needed
    df = df.drop(columns=['Average_Height'])

    # Loop through each combination, filter the DataFrame, save to a CSV file, and print the average height
    for gender, sport in combinations:
        handle_combination(gender=gender, sport=sport, df=df)

# Run the main function
if __name__ == '__main__':
    main()