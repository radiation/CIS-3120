import calendar

def read_original_date() -> str:
    date: str = input("Enter the date in the format MM/DD/YY: ")
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
    print(f"American format: {calendar.month_name[month]}/{day}, {get_four_digit_year(year)}")
    print(f"Full format: {month}-{0 if day.length==1 else ""}{day}-{year}")
    print(f"ISO format: {get_four_digit_year(year)}-{month}-{day}")

def get_four_digit_year(two_digit_year):
    if 0 <= two_digit_year <= 49:
        return 2000 + two_digit_year
    elif 50 <= two_digit_year <= 99:
        return 1900 + two_digit_year
    else:
        raise ValueError("Year must be a two-digit number (00-99).")

def main():
    counter: int = 0
    while date != "quit" and counter <= 10:
        counter += 1
        date: str = read_original_date()
        date_parts: list = break_date(date)
        print_date_three_ways(date_parts)

