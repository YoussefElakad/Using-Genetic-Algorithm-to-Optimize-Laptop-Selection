import requests
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

url = "https://www.sigma-computer.com/search?search=laptop&submit_search=&route=product%2Fsearch"

# Send a GET request to the URL
response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

laptop_items = soup.find_all("div", {"class": "product-layout"})

models = []
prices = []

# Iterate over each laptop item and extract the model and price
for item in laptop_items:
    model_tag = item.find("div", {"class": "caption"}).find("h4")
    model = model_tag.text.strip() if model_tag else "N/A"
    
    price_tag = item.find("p", {"class": "price"})
    price = price_tag.text.strip() if price_tag else "N/A"
    
    models.append(model)
    prices.append(price)

data = {"Model": models, "Price": prices}
df = pd.DataFrame(data)

# Print the data before saving to the Excel file
print("Data to be saved:")
print(df)


filename = r"E:\college\4\1\gp\data\laptops_sigma.xlsx"
df.to_excel(filename, index=False, engine='openpyxl')

print("Data saved to", filename)
