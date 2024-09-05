import pandas as pd
import re

x = r"D:\Faculty\Graduation Project\Python Cleaning\Clean.xlsx"
y = r"D:\Faculty\Graduation Project\Python Cleaning"



#Remove non laptop records
df = pd.read_excel(x)
filtered_df = df[df['Model'].str.contains(r'core|intel|amd|ram|ssd|hdd', case=False, na=False)]
filtered_df.to_excel(y + r"\Labs.xlsx", index=False)



#Manufacturer
df = pd.read_excel(x)

# Extract Manufacturer values from the 'Description' column
df['Manufacturer'] = df['Model'].str.split(n=1).str[0]

# Specify the directory for saving the Excel file
output_dir = y

df.to_excel(output_dir + r"\manufacturer.xlsx", index=False)






#CPU
df = pd.read_excel(x)

# Function to extract CPU information
def extract_cpu(description):
    intel_cpu = re.findall(r'i[3|5|7|9]', description, flags=re.IGNORECASE)
    ryzen_cpu = re.findall(r'ryzen™?\s+(\d+)', description, flags=re.IGNORECASE)
    if intel_cpu:
        return  intel_cpu[0]
    elif ryzen_cpu:
        return 'r' + ryzen_cpu[0]
    else:
        return '0'

# Extract CPU values from the 'Description' column
df['CPU'] = df['Model'].apply(extract_cpu)

# Specify the directory for saving the Excel file
output_dir = y

df.to_excel(output_dir + r"\cpu.xlsx", index=False)






#CPU GEN
df = pd.read_excel(x)

# Extract CPU generation values from the 'Description' column using regular expressions
df['CPU Generation'] = df['Model'].str.extract(r'(\d{4,5}(?:[HHSQUG]|(?<=\d)[HQS]))')

# Fill missing values with '0'
df['CPU Generation'].fillna('0', inplace=True)

# Specify the directory for saving the Excel file
output_dir = y

df.to_excel(output_dir + r"\cpu_generation.xlsx", index=False)




#RAM
df = pd.read_excel(x)

# Check if 'GB Ram' exists in 'Description', if yes, extract RAM value

df['Ram'] = df['Model'].str.extract(r'(\d+)(?=GB)')


# Specify the directory for saving the Excel file
output_dir = y

df.to_excel(output_dir + r"\ram.xlsx", index=False)





#Hard
df = pd.read_excel(x)

# Extract Storage type and capacity values from the 'Description' column using regular expressions
df['SSD'] = df['Model'].str.extract(r'(\d{3}|1) ?(?:GB|TB)')
df['HDD'] = df['Model'].str.extract(r'(\d+) ?(?:GB|TB) HDD')

# Fill missing values with 0
df['SSD'].fillna('0', inplace=True)
df['HDD'].fillna('0', inplace=True)

# Specify the directory for saving the Excel file
output_dir = y

df.to_excel(output_dir + r"\storage.xlsx", index=False)




#GPU
df = pd.read_excel(x)

# Define a function to extract GPU information
def extract_gpu(description):
    gpu_match = re.search(r'(?i)\b(gtx|rtx|mx|rx)\s*(\S+)', description.rstrip(',™'))
    if gpu_match:
        return f"{gpu_match.group(1).upper()} {gpu_match.group(2)}"
    else:
        return 'Integrated'


# Extract GPU information from the 'Description' column
df['GPU'] = df['Model'].apply(extract_gpu)

# Specify the directory for saving the Excel file
output_dir = y

df.to_excel(output_dir + r"\gpu_extraction.xlsx", index=False)