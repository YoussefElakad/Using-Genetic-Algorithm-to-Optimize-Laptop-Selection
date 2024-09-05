import requests
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl

page=requests.get("https://www.jumia.com.eg/catalog/?q=laptop&viewType=list&page=11#catalog-listing")
soup=BeautifulSoup(page.content,"lxml")
laptops=soup.find_all("h3",{"class":"name"})
price=soup.find_all("div",{"class":"prc"})

laptops_list=[]
price_list=[]

for i in range (len(laptops)):
    laptops_list.append(laptops[i].text.strip())
    price_list.append(price[i].text.strip())


data = {"Model": laptops_list, "Price": price_list}
df = pd.DataFrame(data)

# Print the data before saving to the Excel file
print("Data to be saved:")
print(df)

filename = r"E:\college\4\1\gp\data\jumia11.xlsx"
df.to_excel(filename, index=False, engine='openpyxl')

print("Data saved to", filename)   
 