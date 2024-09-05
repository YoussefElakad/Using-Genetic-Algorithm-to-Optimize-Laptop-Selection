""" import csv
import requests
from bs4 import BeautifulSoup
from itertools import zip_longest

name = []
Price = []

result = requests.get("https://egyptlaptop.com/laptops-and-notebooks")

src = result.content

soup = BeautifulSoup(src, "lxml")

# Name ,Price
# Name=soup.find_all("div",{"class":"ut2-gl__name"})
Name = soup.find_all("a", {"class": "product-title"})
price = soup.find_all("span", {"class":"ty-price"})

for i in range(len(price)):
    name.append(Name[i].text)
    Price.append(price[i].text)

# Use zip_longest to handle uneven lengths of Name and Price
file_list=[name,Price]
exported= zip_longest(*file_list)               

# Assuming you want to write the data to a CSV file
with open('D:\WORK\Web Scraping\EgyptLabtop.csv', 'w',newline='',encoding='utf-8') as file:
    wr = csv.writer(file)
    # Write header
    wr.writerow(['Name', 'Price'])
    # Write data
    wr.writerows(exported) 

 """

import csv
import requests
from bs4 import BeautifulSoup
from itertools import zip_longest

name = []
Price = []

page_number = 1

while True:
    url = f"https://egyptlaptop.com/laptops-and-notebooks?page={page_number}"
    result = requests.get(url)
    src = result.content
    soup = BeautifulSoup(src, "lxml")

    # Check if there are no products on the page, indicating the last page
    if not soup.find("a", {"class": "product-title"}):
        break

    # Name ,Price
    Name = soup.find_all("a", {"class": "product-title"})
    price = soup.find_all("span", {"class": "ty-price"})

    for i in range(len(price)):
        name.append(Name[i].text)
        cleaned_price = price[i].text.replace('\xa0', ' ')
        Price.append(cleaned_price)

    # Move to the next page
    page_number += 1

# Use zip_longest to handle uneven lengths of Name and Price
rows = zip_longest(name, Price)

# Assuming you want to write the data to a CSV file
with open('D:\WORK\Web Scraping\EgyptLaptob.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Name', 'Price'])
    # Write data
    writer.writerows(rows)
