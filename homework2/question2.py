import random
import requests
import pandas as pd

def get_comic(num: int = None) -> dict:
    url = f"https://xkcd.com/{num}/info.0.json" if (num or num == 1) \
        else "https://xkcd.com/info.0.json"
    response = requests.get(url)
    response.raise_for_status()
    comic = response.json()
    return comic

def main():
    comic: dict = get_comic()
    print(f"{comic['num']} comics so far!")

    # Generate 10 random numbers between 1 and current comic number
    comic_num = comic['num']
    random_comic_nums = random.sample(range(1, comic_num + 1), 10)

    # Initialize an empty list to store comic data
    comic_data = []

    # Get 10 random comics
    for i in random_comic_nums:
        comic_data.append(get_comic(i))

    # Drop the transcript column and save the data to a CSV file
    df = pd.DataFrame(comic_data).drop(columns=['transcript'])
    df.to_csv('random_comics.csv', index=False)

if __name__ == "__main__":
    main()