import pandas as pd
import sys

#Encoded Data Frame
df = pd.read_excel('D:\Faculty\Graduation Project\Project Implementation\Encoded.xlsx', sheet_name='Sheet1')
#user inputs
Usage = input("Enter Laptop Usage: ")
Budget = float(input("Enter Your Budget: "))


#Weighting and Constraints
if Usage == 'gaming':
    W1 = 0.7
    W2 = 0.3
    WFF = [0.2,0.1,0.1,0.2,0.05,0.35]
    C = {'CPU' :15 ,
        'PowerEfficiency':3 ,
        'Ram':8 ,
        'SSD':2 ,
        'HDD':0 ,
        'GPU':9}
elif Usage == 'coding':
    W1 = 0.3
    W2 = 0.7
    WFF = [0.35,0.1,0.2,0.2,0.05,0.1]
    C = {'CPU' :10 ,
        'PowerEfficiency':0 ,
        'Ram':4 ,
        'SSD':1 ,
        'HDD':0 ,
        'GPU':0}
elif Usage == 'content':
    W1 = 0.6
    W2 = 0.4
    WFF = [0.2,0.1,0.1,0.1,0.1,0.4]
    C = {'CPU' :15 ,
        'PowerEfficiency':0 ,
        'Ram':8 ,
        'SSD':0 ,
        'HDD':0 ,
        'GPU':7}
elif Usage == 'office':
    W1 = 0.2
    W2 = 0.8
    WFF = [0.3,0.1,0.2,0.2,0.1,0.1]
    C = {'CPU' :7 ,
        'PowerEfficiency':0 ,
        'Ram':4 ,
        'SSD':0 ,
        'HDD':0 ,
        'GPU':0}
elif Usage == 'media':
    W1 = 0.2
    W2 = 0.8
    WFF = [0.3,0.1,0.2,0.1,0.2,0.1]
    C = {'CPU' :0 ,
        'PowerEfficiency':0 ,
        'Ram':4 ,
        'SSD':1 ,
        'HDD':1 ,
        'GPU':0}
elif Usage == 'student':
    W1 = 0.2
    W2 = 0.8
    WFF = [0.3,0.2,0.1,0.1,0.2,0.1]
    C = {'CPU' :0 ,
        'PowerEfficiency':0 ,
        'Ram':0 ,
        'SSD':0 ,
        'HDD':0 ,
        'GPU':0}
else:
    W1 = 0.5
    W2 = 0.5
    WFF = [1/6,1/6,1/6,1/6,1/6,1/6]
    C = {'CPU' :0 ,
        'PowerEfficiency':0 ,
        'Ram':0 ,
        'SSD':0 ,
        'HDD':0 ,
        'GPU':0}



# Function to check if a row satisfies the constraints
def check_constraints(row, constraints):
    return (row['CPU'] >= constraints['CPU'] and
            row['PowerEfficiency'] >= constraints['PowerEfficiency'] and
            row['Ram'] >= constraints['Ram'] and
            row['SSD'] >= constraints['SSD'] and
            row['HDD'] >= constraints['HDD'] and
            row['GPU'] >= constraints['GPU'] and
            row['Price'] <= Budget/10000)

# List to hold rows that meet the constraints
filtered_rows = []

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    if check_constraints(row, C):
        filtered_rows.append(row)

# Create a new DataFrame from the filtered rows
filtered_df = pd.DataFrame(filtered_rows)

#Check if empty
if filtered_df.empty:
    print("No Laptop Found")
    sys.exit()

# Objective function
def objective_function(row):
    return W1*(WFF[0]*row['CPU'] + WFF[1]*row['PowerEfficiency'] + WFF[2]*row['Ram'] + WFF[3]*row['SSD'] + WFF[4]*row['HDD'] + WFF[5]*row['GPU'])-W2*(row['Price'])

# Apply objective function to each row
filtered_df['F'] = filtered_df.apply(objective_function, axis=1)
print("DataFrame with Objective Function:")
print(filtered_df)

#Choosing Optimal
optimal = filtered_df['F'].max()
Optimal_Row = filtered_df[filtered_df['F'] == optimal][['laptop', 'Manufacturer', 'Price', 'store']]

print("\nOptimal Laptop")
Optimal_Row['Price'] = int(Optimal_Row['Price'] * 10000)
print(Optimal_Row)