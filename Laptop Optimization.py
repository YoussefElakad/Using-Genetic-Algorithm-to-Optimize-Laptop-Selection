from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

#Flask Server
#############################################################################################
app = Flask(__name__, template_folder='templates')

#Creating a server for the HTML Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find-laptops', methods=['POST'])
def find_laptops():
    selection_type = request.form.get('selectionType')
    budget = request.form.get('budget')
    usage = request.form.get('usage')

    if selection_type == 'customize':
        results_df = process_laptop_data_Custom(budget, usage)
    else:
        results_df = process_laptop_data_Existing(budget, usage)

    results_html = results_df.to_html(classes='dataframe', header=True, index=False)
    
    return render_template('results.html', results_html=results_html)
#############################################################################################


#Custom laptop function (Genetic Algorithm)
#############################################################################################
def process_laptop_data_Custom(Budget, Usage):
    Budget = int(Budget)
    #Insert Custom.py here
    #############################################################################################

    #Initialization
    ########################################################################
    #Encoded Data Frame
    Reference_df =  pd.read_excel('D:\Faculty\Graduation Project\Project Implementation\Encoded.xlsx', sheet_name='Sheet1')
    df = pd.read_excel('D:\Faculty\Graduation Project\Project Implementation\Encoded.xlsx', sheet_name='Sheet1',usecols=['CPU','PowerEfficiency','Ram','SSD','HDD','GPU'])

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


    #Choosing random initial population from encoded database
    def initPop(df,npop):
        pop = df.sample(npop)
        return pop
    ########################################################################


    #Fitness Evaluation
    ########################################################################
    #Using KNN to predect population price
    def PredictPrice(input_df, Reference_df, n_neighbors):
        feature_columns = ['CPU','PowerEfficiency','Ram','SSD','HDD','GPU']
        target_column = 'Price'

        # Extract features and target from the reference DataFrame
        X_train = Reference_df[feature_columns]
        y_train = Reference_df[target_column]

        # Initialize and fit the kNN regressor
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        # Predict prices for the input DataFrame
        predicted_prices = knn.predict(input_df)

        # Add the predicted prices as a new column to the input DataFrame
        input_df['Price'] = predicted_prices

        return input_df


    # Fitness function
    def fitness_evaluation(popdf,C):
        X = PredictPrice(popdf, Reference_df, n_neighbors=3)

        F = W1*(WFF[0]*X['CPU'] + WFF[1]*X['PowerEfficiency'] + WFF[2]*X['Ram'] + WFF[3]*X['SSD'] + WFF[4]*X['HDD'] + WFF[5]*X['GPU'])-W2*(X['Price'])
        constraints = pd.DataFrame({
            'CPU': X['CPU'] - C['CPU'],
            'PowerEfficiency': X['PowerEfficiency'] - C['PowerEfficiency'],
            'Ram': X['Ram'] - C['Ram'],
            'SSD': X['SSD'] - C['SSD'],
            'HDD': X['HDD'] - C['HDD'],
            'GPU': X['GPU'] - C['GPU'],
            'Price': (Budget/10000 - X['Price'])*5,
        })
        penalties = constraints.map(lambda x: min(0, x))
        penalty = penalties.sum(axis=1)

        X['F'] = F + penalty
        return X


    #Elitism
    def Elite(popdf):
        popdf['F'] = popdf['F'].astype(float)
        return popdf.nlargest(1,'F')
    ########################################################################


    #Selection
    ########################################################################
    #Linear Ranking
    def LinRank(popdf):
        sp = 1.5

        #Ranking
        popdf['Rank'] = popdf['F'].rank(method='min')
        #Fitness Ranking
        popdf['RankFit'] = (2 - sp) + 2 * (sp - 1) * (popdf['Rank'] - 1)/ (popdf.shape[0] - 1)
        #Probability
        popdf['Prob'] = popdf['RankFit']/sum(popdf['RankFit'])
        #Cumulative Probability
        popdf['Cumprob'] = popdf['Prob'].cumsum()

        return popdf


    #Roulette Wheel
    def rouletteWheel(popdf):
        crosscols = ['CPU','PowerEfficiency','Ram','SSD','HDD','GPU']
        otherscols = ['CPU','PowerEfficiency','Ram','SSD','HDD','GPU','Price','F']
        twoparentsrs = []

        #If only one individual left return it
        if popdf.shape[0] == 1:
            return popdf[crosscols],pd.DataFrame()

        #Choosing two rows based on random values
        for i in range(2):
            r = np.random.random()
            for index, j in popdf.iterrows():
                if r <= j['Cumprob']:
                    twoparentsrs.append(j[crosscols])
                    break
        
        twoparents = pd.DataFrame(twoparentsrs)

        #Checking if both parent uinque
        if twoparents.nunique().eq(1).all():
            twoparents = twoparents.iloc[[0]]

        #returning population without twoparents
        others = popdf[otherscols]
        mask = ~others.index.isin(twoparents.index)
        others = others[mask]

        return twoparents.astype(int),others


    #Recombination
    ########################################################################
    #Crossover
    def Cross(popdf,pcross):
        twoparents,others = rouletteWheel(popdf)

        #Only one individual
        if twoparents.shape[0] == 1:
            return twoparents,others

        Parent1 = twoparents.iloc[0]
        Parent2 = twoparents.iloc[1]
        
        #Perform Crossover
        if np.random.random() < pcross:
            crossover_point = np.random.randint(1, 5)
            Child1 = pd.DataFrame(pd.concat([Parent1.iloc[:crossover_point], Parent2.iloc[crossover_point:]])).T
            Child2 = pd.DataFrame(pd.concat([Parent2.iloc[:crossover_point], Parent1.iloc[crossover_point:]])).T

            twoChildren = pd.concat([Child1, Child2], axis=0).reset_index(drop=True)
        #Return Two Parents
        else:
            twoChildren = twoparents

        return twoChildren,others


    #Mutation
    def Mutate(popdf, pmute):
        for index, row in popdf.iterrows():
            for column in popdf.columns:
                if np.random.random() < pmute:
                    if column == 'CPU':
                        if popdf.at[index, column] == 44:
                            popdf.at[index, column] = row[column] - 1
                        else:
                            popdf.at[index, column] = row[column] + 1
                    if column == 'PowerEfficiency':
                        if popdf.at[index, column] == 4:
                            popdf.at[index, column] = row[column] - 1
                        else:
                            popdf.at[index, column] = row[column] + 1
                    if column == 'Ram':
                        if popdf.at[index, column] == 16:
                            popdf.at[index, column] = row[column] + 16
                        elif popdf.at[index, column] == 32:
                            popdf.at[index, column] = row[column] - 16
                        elif popdf.at[index, column] < 4:
                            popdf.at[index, column] = row[column] + 1
                        elif popdf.at[index, column] >= 8:
                            popdf.at[index, column] = row[column] + 4
                        elif popdf.at[index, column] >= 4:
                            popdf.at[index, column] = row[column] + 2
                    if column == 'SSD':
                        if popdf.at[index, column] == 7:
                            popdf.at[index, column] = row[column] - 1
                        else:
                            popdf.at[index, column] = row[column] + 1
                    if column == 'HDD':
                        if popdf.at[index, column] == 3:
                            popdf.at[index, column] = row[column] - 1
                        else:
                            popdf.at[index, column] = row[column] + 1
                    if column == 'GPU':
                        if popdf.at[index, column] == 23:
                            popdf.at[index, column] = row[column] - 1
                        else:
                            popdf.at[index, column] = row[column] + 1

        return popdf


    #Decode
    ########################################################################
    #Decode
    def Decode(Findf):
        #Values
        CPUArr = [
        "Celeron", "Pentium", "M3", "Atom", "A4", "A6", "i5 6th gen", "i3 8th gen", "r3 3rd gen", "i5 8th gen",
        "i3 10th gen", "r5 3rd gen", "i3 11th gen", "i5 10th gen", "i7 7th gen", "r3 5th gen", "i5 11th gen", "i7 8th gen", "i7 9th gen", "i7 10th gen",
        "i3 12th gen", "r5 4th gen", "Z1", "i5 12th gen", "i7 11th gen", "r5 5th gen", "r7 3rd gen", "r7 4th gen", "i5 13th gen", "r3 7th gen",
        "r5 6th gen", "i7 12th gen", "r7 5th gen", "r5 7th gen", "r7 6th gen", "i9 10th gen", "r7 7th gen", "i7 13th gen", "r9 5th gen", "i9 11th gen",
        "i9 12th gen", "r9 6th gen", "i9 13th gen", "r9 7th gen"]
        PowArr = ["U","G" ,"P" ,"H"]
        SSDArr = [64,128,256,512,1000,2000,3000]
        HDDArr = [500,1000,2000]
        GPUArr = ["Integrated", "MX", "GTX 1050", "RX 560", "RX 640", "GTX 1650", "RX 6500", "RTX 2050", "GTX 1660", "RTX 3050",
        "RX 6600", "RTX 2060", "RTX 4050", "RTX 2070", "RTX 3060", "RTX 2080", "RX 6700", "RTX 4060", "RTX 3070", "RTX 4070",
        "RTX 3080", "RTX 4080", "RTX 4090"]

        #Mapping
        cpu_mapping = {i + 1: CPUArr[i] for i in range(len(CPUArr))}
        power_mapping = {i + 1: PowArr[i] for i in range(len(PowArr))}
        ssd_mapping = {i + 1: SSDArr[i] for i in range(len(SSDArr))}
        hdd_mapping = {i + 1: HDDArr[i] for i in range(len(HDDArr))}
        gpu_mapping = {i + 1: GPUArr[i] for i in range(len(GPUArr))}

        #Decoding
        Findf['CPU'] = Findf['CPU'].map(cpu_mapping)
        Findf['PowerEfficiency'] = Findf['PowerEfficiency'].map(power_mapping)
        Findf['SSD'] = Findf['SSD'].map(ssd_mapping)
        Findf['HDD'] = Findf['HDD'].map(hdd_mapping)
        Findf['GPU'] = Findf['GPU'].map(gpu_mapping)

        for index, row in Findf.iterrows():
            Findf.at[index, 'Ram'] = row['Ram'] * 2
            Findf.at[index, 'Price'] = row['Price'] * 10000

        Findf['Price'] = Findf['Price'].astype(int)
        Findf['SSD'] = pd.to_numeric(Findf['SSD'], errors='coerce').astype('Int64')
        Findf['HDD'] = pd.to_numeric(Findf['HDD'], errors='coerce').astype('Int64')

        return Findf


    #Main Function
    ########################################################################
    # Parameters
    n_runs = 10
    npop = 10
    ngen = 50
    pcross = 0.6
    pmute = 0.05


    #Main
    def Main(df):
        #initial
        popdf = initPop(df,npop)
        #Empty Elite
        El = pd.DataFrame(columns=['CPU','PowerEfficiency','Ram','SSD','HDD','GPU','Price','F'])
        #Temp Dataframe
        tempdf = pd.DataFrame(columns=['CPU','PowerEfficiency','Ram','SSD','HDD','GPU'])

        #Generations
        for i in range(ngen):
            #Fitness Evaluation
            popdf = fitness_evaluation(popdf,C)
            #Insert Elite of current generation
            El = pd.concat([El, Elite(popdf)], ignore_index=True)
            #Crossover
            while not popdf.empty:
                popdf = LinRank(popdf)
                Offsrping,popdf = Cross(popdf,pcross)
                tempdf = pd.concat([tempdf, Offsrping], ignore_index=True)
            popdf = tempdf
            tempdf = pd.DataFrame(columns=['CPU','PowerEfficiency','Ram','SSD','HDD','GPU'])

            #Mutation
            popdf = Mutate(popdf,pmute)
        return El


    #Outputs
    ########################################################################
    Generation_Elite = pd.DataFrame(columns=['CPU','PowerEfficiency','Ram','SSD','HDD','GPU','Price','F'])
    Final_Elite = pd.DataFrame(columns=['CPU','PowerEfficiency','Ram','SSD','HDD','GPU','Price','F'])

    for i in range (n_runs):
        Generation_Elite = pd.concat([Generation_Elite, Elite(Main(df))], ignore_index=True)

    Final_Elite = Generation_Elite.nlargest(3,'F')
    Final_Elite.drop(columns=['F'], inplace=True)


    Final_Elite = Decode(Final_Elite)

    #############################################################################################
    


    return Final_Elite
#############################################################################################


#Existing laptop function
#############################################################################################
def process_laptop_data_Existing(Budget, Usage):
    Budget =int(Budget)
    #Insert Existing.py here
    #############################################################################################
    #Encoded Data Frame
    df = pd.read_excel('D:\Faculty\Graduation Project\Project Implementation\Encoded.xlsx', sheet_name='Sheet1')

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
        emptdata = {'Satus':['No Laptop Found, Try choosing another preference or increasing your budget!']}
        return pd.DataFrame(emptdata)

    # Objective function
    def objective_function(row):
        return W1*(WFF[0]*row['CPU'] + WFF[1]*row['PowerEfficiency'] + WFF[2]*row['Ram'] + WFF[3]*row['SSD'] + WFF[4]*row['HDD'] + WFF[5]*row['GPU'])-W2*(row['Price'])

    # Apply objective function to each row
    filtered_df['F'] = filtered_df.apply(objective_function, axis=1)

    #Choosing Optimal
    optimal = filtered_df['F'].max()
    Optimal_Row = filtered_df[filtered_df['F'] == optimal][['laptop', 'Manufacturer', 'Price', 'store']]

    Optimal_Row['Price'] = Optimal_Row['Price'] * 10000
    Optimal_Row['Price'] = Optimal_Row['Price'].astype(int)
    
    #############################################################################################


    return Optimal_Row
#############################################################################################


if __name__ == '__main__':
    app.run(debug=True)