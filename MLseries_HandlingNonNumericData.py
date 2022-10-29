# TITANIC DATASET: https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbUU3UTZzOUx2SmVsNmdrU25ZYi1HU1ZtQ0d4Z3xBQ3Jtc0tubm0tRUlweWRKWGlXMTJyTy1ZZ2ZCVVRIVXdQVzlma29hSG83LXN1VFY1dXcwajFFeFpPTnJNWDFHNWQ2Njc3RTkwQVBvejJNY2hZOWdWN2JaMlVlUW96ZjRVLWhaQkl4TlpNUld0ZnR2TFd2RHVzUQ&q=https%3A%2F%2Fpythonprogramming.net%2Fstatic%2Fdownloads%2Fmachine-learning-data%2Ftitanic.xls&v=8p6XaQSIFpY
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

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

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
correct_dead = 0
correct_alive = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    print('pi: ', prediction[0])
    print('yi: ', y[i])
    print('___________________')
    if prediction[0] == y[i]:
        correct += 1
        if y[i] == 0:
            correct_dead += 1
        else:
            correct_alive += 1


print(correct/len(X))
print(correct_dead/len(y*(y==0)))
print(correct_alive/len(y*(y==1)))


