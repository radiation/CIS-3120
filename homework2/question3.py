import calendar  # For converting month number to month name
from datetime import \
    datetime  # Bonus: For converting the last 100 years to four-digit years

import inflect  # Bonus: For converting cardinal numbers to ordinal numbers

# Create an inflect engine
p = inflect.engine()

# Get current year outside of a function so we don't have to call it every time
# Divide by 100 to get the last two digits of the year
current_year: int = datetime.now().year % 100

def read_original_date() -> str:
    date: str = input("Enter the date in the format MM/DD/YY or \"q\" to quit: ")
    return date

def break_date(date: str) -> list:
    date_parts: list = date.split('/')
    print(f"Original date: {date}")
    return date_parts

def print_date_three_ways(date_parts: list):
    month: str = date_parts[0]
    day: str = date_parts[1]
    year: str = date_parts[2]
    print(f"European format: {day}-{month}-{year}")
    print(f"American format: {calendar.month_name[int(month)]} {p.ordinal(day)}, {get_four_digit_year(year)}")

    # Pad with zeroes if necessary
    month = month.zfill(2)
    day = day.zfill(2)
    print(f"Full format: {month}-{day}-{year.zfill(2)}")

    # Bonus: ISO8601 format, which is objectively the best because it can be alphabetized
    print(f"ISO8601 format: {get_four_digit_year(year)}-{month}-{day}")

# Convert a two-digit year to a four-digit year
def get_four_digit_year(two_digit_year: str) -> int:
    two_digit_year: int = int(two_digit_year)
    if two_digit_year <= current_year:
        return 2000 + two_digit_year
    elif two_digit_year <= 99:
        return 1900 + two_digit_year
    else:
        raise ValueError("Year must be a two-digit number (00-99).")

def main():
    counter: int = 0
    date: str = ""
    while counter <= 10:
        counter += 1
        date: str = read_original_date()
        if date.lower() == "q":
            break
        date_parts: list = break_date(date)
        print_date_three_ways(date_parts)

if __name__ == "__main__":
    main()