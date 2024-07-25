import requests
from bs4 import BeautifulSoup

url = 'https://avinashjairam.github.io/syllabus'

headers = {
    'User-Agent': 'Mozilla/5.0'
}

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.content, 'html.parser')

print(soup.find('p', id='email').text.strip())