import requests
import pandas as pd
from bs4 import BeautifulSoup

# This function takes a DataFrame and a list of column names as input
def print_stat_summary(df, stats):

    # Suppress annoying SettingWithCopyWarning
    with pd.option_context('mode.chained_assignment', None):
        # Convert each column to a numeric type so we can operate on them
        df[stats] = df[stats].apply(pd.to_numeric, errors='coerce')

    # List comprehension to filter out non-numeric columns
    valid_stats = [stat for stat in stats if stat in df.columns and pd.api.types.is_numeric_dtype(df[stat])]

    if valid_stats:
        # Calculate & print the combined statistics for all valid columns
        summary = df[valid_stats].describe(percentiles=[0.25, 0.5, 0.75])
        print(summary.round(2))
    else:
        print("None of the specified columns are numeric or available in the DataFrame.")

# Function to parse a single BeautifulSoup row
def parse_row(row):
    data = {}
    for cell in row.find_all(['th', 'td']):
        stat_name = cell['data-stat']
        if stat_name == '...' or stat_name is None:
            continue
        value = cell.get_text(strip=True)
        data[stat_name] = value
    return data

# Custom function to match class names that contain 'per_game'; tag is implicitly passed by BeautifulSoup
def has_per_game_id(tag) -> bool:
    return tag.has_attr('id') and 'per_game' in tag['id']

# Main function, obviously
def main():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }
    url = "https://www.basketball-reference.com/players/j/jamesle01.html"
    print(f"\nFetching {url}...\n")

    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, 'html.parser')
    rows = soup.find_all(has_per_game_id)

    # Build a list of dictionaries from the parsed rows
    data_list = [parse_row(row) for row in rows]

    # There are A LOT OF columns in the table, so we need to see them all
    pd.set_option('display.max_columns', None)

    # Convert the list of dictionaries to a DataFrame & filter out rows that don't contain a season
    df = pd.DataFrame(data_list)
    filtered_df = df[df['season'].str.contains(r'\d{4}-\d{2}', na=False)]
    filtered_df.to_csv('lebron_james.csv', index=False)

    # Specify the columns we want to summarize and print the summary; let's use triple-double stats
    stat_columns = ['pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g']
    print_stat_summary(filtered_df, stat_columns)

if __name__ == "__main__":
    main()
