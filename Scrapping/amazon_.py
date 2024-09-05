from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import pandas as pd
import openpyxl

laptops_list = []
special_price_list = []
brand_name_list = []
model_name_list = []
screen_size_list = []
hard_disk_size_list = []
cpu_model_family_list = []

url = "https://www.amazon.eg/s?k=laptop&i=electronics&rh=n%3A18018102031%2Cp_89%3AASUS%7CAcer%7CApple%7CDell%7CHP%7CHUAWEI%7CLenovo%7CMSI%7CMicrosoft%7CSAMSUNG&dc&language=en&ds=v1%3A5c1xzdbNXJHZ8IcfguWVQzNTRN7hOR5paygPlCNCezM&crid=1G5CFEQBDY5HS&qid=1701902137&rnid=22541269031&sprefix=lap%2Caps%2C1015&ref=sr_nr_p_89_30"

# Use a headless browser (invisible) to scrape dynamic content
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(options=options)
driver.get(url)

# Wait for some time to let the dynamic content load (you may need to adjust the time)
driver.implicitly_wait(10)

page_source = driver.page_source

driver.quit()

soup = BeautifulSoup(page_source, "lxml")

link_of_product = soup.find_all("a", {"class": "a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal"})

for link in link_of_product:
    # Get the absolute URL by joining the base URL with the relative URL from the link
    absolute_url = urljoin(url, link["href"])
    
    result = requests.get(absolute_url)
    
    # Parse the content of the product page
    soup_product = BeautifulSoup(result.content, "lxml")
    
    laptops = soup_product.find_all("span", {"id": "productTitle"})
    special_price = soup_product.find_all("span", {"class": "a-price-whole"})
    brand_name_element = soup_product.find("tr", {"class": "a-spacing-small po-brand"})
    if brand_name_element:
      brand_name_element = brand_name_element.find("span", {"class": "a-size-base po-break-word"})
    model_name_element = soup_product.find("tr", {"class": "a-spacing-small po-model_name"})
    if model_name_element:
      model_name_element = model_name_element.find("span", {"class": "a-size-base po-break-word"})

    screen_size_element = soup_product.find("tr", {"class": "a-spacing-small po-display.size"})
    if screen_size_element:
      screen_size_element = screen_size_element.find("span", {"class": "a-size-base po-break-word"})

    hard_disk_size_element = soup_product.find("tr", {"class": "a-spacing-small po-hard_disk.size"})
    if hard_disk_size_element:
      hard_disk_size_element = hard_disk_size_element.find("span", {"class": "a-size-base po-break-word"})

    cpu_model_family_element = soup_product.find("tr", {"class": "a-spacing-small po-cpu_model.family"})
    if cpu_model_family_element:
      cpu_model_family_element = cpu_model_family_element.find("span", {"class": "a-size-base po-break-word"})

    
    laptops_list.append(laptops[0].text.strip() if laptops else "N/A")
    special_price_list.append(special_price[0].text.strip() if special_price else "N/A")
    brand_name_list.append(brand_name_element.text.strip() if brand_name_element else "N/A")
    model_name_list.append(model_name_element.text.strip() if model_name_element else "N/A")
    screen_size_list.append(screen_size_element.text.strip() if screen_size_element else "N/A")
    hard_disk_size_list.append(hard_disk_size_element.text.strip() if hard_disk_size_element else "N/A")
    cpu_model_family_list.append(cpu_model_family_element.text.strip() if cpu_model_family_element else "N/A")

    print("Laptops:", laptops_list[-1])
    print("Price:", special_price_list[-1])
    print("Brand:", brand_name_list[-1])
    print("Model:", model_name_list[-1])
    print("Screen Size:", screen_size_list[-1])
    print("Hard:", hard_disk_size_list[-1])
    print("CPU:", cpu_model_family_list[-1])
    print("\n")  
data = {
    'Laptops': laptops_list,
    'Price': special_price_list,
    'Brand': brand_name_list,
    'Model': model_name_list,
    'Screen Size': screen_size_list,
    'Hard Disk': hard_disk_size_list,
    'CPU': cpu_model_family_list
}


df = pd.DataFrame(data)

df.to_excel(r'E:\college\4\1\gp\data\amazon_data.xlsx', index=False)

