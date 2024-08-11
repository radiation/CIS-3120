import requests
import pandas as pd
from bs4 import BeautifulSoup

def print_stat_summary(df, stat):
    # Temporarily suppress the SettingWithCopyWarning
    with pd.option_context('mode.chained_assignment', None):
        # Convert the column to a numeric type
        df[stat] = pd.to_numeric(df[stat], errors='coerce')

    if stat in df.columns and pd.api.types.is_numeric_dtype(df[stat]):
        # Calculate & print the statistics
        summary = df[stat].describe(percentiles=[0.25, 0.5, 0.75])  
        print(summary)

# Function to parse a single row
def parse_row(row):
    data = {}
    for cell in row.find_all(['th', 'td']):
        stat_name = cell['data-stat']


        if stat_name == '...' or stat_name is None:
            continue
        value = cell.get_text(strip=True)
        data[stat_name] = value
    return data

# Custom function to match class names that contain 'emotion-srm-'
def has_per_game_id(tag) -> bool:
    return tag.has_attr('id') and 'per_game' in tag['id']

def main():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }
    url = "https://www.basketball-reference.com/players/j/jamesle01.html"
    print(f"Fetching {url}...")

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')
    rows = soup.find_all(has_per_game_id)

    # Initialize an empty list to store row data
    data_list = []
    pattern = r'\d{4}-\d{2}'

    for row in rows:
        row_data = parse_row(row)
        data_list.append(row_data)

    pd.set_option('display.max_columns', None)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)
    filtered_df = df[df['season'].str.contains(pattern, na=False)]
    filtered_df.to_csv('lebron_james.csv', index=False)

    stat_columns = ['pts_per_g', 'trb_per_g']
    for stat in stat_columns:
        print_stat_summary(filtered_df, stat)

    # Display the resulting DataFrame
    # print(filtered_df)

if __name__ == "__main__":
    main()
