# TITANIC DATASET: https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbUU3UTZzOUx2SmVsNmdrU25ZYi1HU1ZtQ0d4Z3xBQ3Jtc0tubm0tRUlweWRKWGlXMTJyTy1ZZ2ZCVVRIVXdQVzlma29hSG83LXN1VFY1dXcwajFFeFpPTnJNWDFHNWQ2Njc3RTkwQVBvejJNY2hZOWdWN2JaMlVlUW96ZjRVLWhaQkl4TlpNUld0ZnR2TFd2RHVzUQ&q=https%3A%2F%2Fpythonprogramming.net%2Fstatic%2Fdownloads%2Fmachine-learning-data%2Ftitanic.xls&v=8p6XaQSIFpY
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

pd.set_option('display.max_columns', 20)

'''
TITANIC DATASET FORMATTING 
Pclass      Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival    Survival (0 = no; 1 = yes)
name        Name
sex         Sex
age         Age
sibsp       Number of Siblings/Spouses Aboard
parch       Number of Parents/Children Abroad
ticket      Ticket Number
fare        Passenger Fare (in British pound)
cabin       Cabin
embarked    Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat        Lifeboat
body        Body Identification Number
home.dest   Home/Destination
'''

df = pd.read_excel('MLseries_dataset_titanic.xls')
original_df = pd.DataFrame.copy(df) # original_df = df, would apply any alteration to original_df also to df! So we have to copy like this!
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
#print(df.head())

df.drop(['boat', 'sex'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print(temp_df.describe())

print(survival_rates)




