import requests
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl

page=requests.get("https://dream2000.com/en/laptop-notebook.html?_=1701814343558&cat=557%2C635%2C636%2C637%2C638&p=2&product_list_limit=100")
soup=BeautifulSoup(page.content,"lxml")
laptops=soup.find_all("a",{"class":"product-item-link"})
special_price=soup.find_all("span",{"class":"special-price"})
old_price=soup.find_all("span",{"data-price-type":"oldPrice"})
laptops1=[]
s_price1=[]
o_price=[]

for i in range (len(laptops)):
    laptops1.append(laptops[i].text.strip())
    s_price1.append(special_price[i].text.strip())
    o_price.append(old_price[i].text.strip())


data = {"Model": laptops1, "Price": s_price1,"old price":o_price}
df = pd.DataFrame(data)

# Print the data before saving to the Excel file
print("Data to be saved:")
print(df)

filename = r"E:\college\4\1\gp\data\laptops_2000dreams_2.xlsx"
df.to_excel(filename, index=False, engine='openpyxl')

print("Data saved to", filename)   
 