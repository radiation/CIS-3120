from enum import Enum
import requests
from bs4 import BeautifulSoup

# Global vars
name_width: int = 25
height_width: int = 5
line_break: str = '-'*(name_width+height_width)

# Enum for sports to prevent data entry errors
class Sport(Enum):
    SWIMMING = 'Swimming'
    VOLLEYBALL = 'Volleyball'

# Enum for genders
class Gender(Enum):
    MALE = 'M'
    FEMALE = 'F'

# Enum for schools
class School(Enum):
    BARUCH = 'Baruch College'


# Use a class for Athletes in case we want to compare other attributes in the future
class Athlete:
    def __init__(self, name, gender, sport, height, school):
        self.name: str = name
        self.gender: Gender = gender
        self.sport: Sport = sport
        self.height: float = height
        self.school: School = school
    
    def __str__(self):
        return f'{self.name}: {self.gender} {self.sport} @ {self.school}'


# Use a class for SportsSource to encapsulate the data
class SportsProgram:
    def __init__(self, url, school, gender, sport):
        self.url = url
        self.school = school
        self.gender = gender
        self.sport = sport

        # Set headers to avoid 403 Forbidden error
        self.headers = {
            'User-Agent': 'Mozilla/5.0'
        }

        # Pull data
        self.response = requests.get(self.url, headers=self.headers)
        self.soup = BeautifulSoup(self.response.content, 'html.parser')
        names = self.soup.find_all('td', class_='sidearm-table-player-name')
        heights = self.soup.find_all('td', class_='height')

        # Create list of Athlete objects
        self.player_list: list[Athlete] = []
        for name, height in zip(names, heights):
            self.player_list.append(
                Athlete(
                    name = name.get_text(strip=True), 
                    gender = self.gender, 
                    sport = self.sport, 
                    height = float(height['data-sort']), 
                    school = self.school
                )
            )

    # Return the players for a program
    def get_players(self):
        return self.player_list
    

# Main function
def main():
    school_programs: list[SportsProgram] = [
        SportsProgram('https://athletics.baruch.cuny.edu/sports/mens-swimming-and-diving/roster?view=2',
                        School('Baruch College'),
                        Gender('M'),
                        Sport('Swimming')),
        SportsProgram('https://athletics.baruch.cuny.edu/sports/mens-volleyball/roster?view=2',
                        School('Baruch College'),
                        Gender('M'),
                        Sport('Volleyball')),
        SportsProgram('https://athletics.baruch.cuny.edu/sports/womens-swimming-and-diving/roster?view=2',
                        School('Baruch College'),
                        Gender('F'),
                        Sport('Swimming')),
        SportsProgram('https://athletics.baruch.cuny.edu/sports/womens-volleyball/roster?view=2',
                        School('Baruch College'),
                        Gender('F'),
                        Sport('Volleyball')),
    ]

    # Iterate through each program and print the average height
    for program in school_programs:
        players = program.get_players()
        print(line_break)
        print(f'{program.sport} - {program.school}, {players[0].gender}')

        # We use a list comprehension because we have player objects instead of heights
        avg_height = sum([int(player.height) for player in players])/len(players)
        print(f'Average height: {round(avg_height, 2)}')
        print(f'From {len(program.get_players())} data points:\n')

        # We can do this because these are small teams;
        # in the real world we'd write them to a CSV or DB
        for player in program.get_players():
            print('{:>{}}{:>{}}'.format(player.name, name_width, int(player.height), height_width))

    print(line_break)

if __name__ == '__main__':
    main()