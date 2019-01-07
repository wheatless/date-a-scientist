import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
import time

#Create your df here:
df = pd.read_csv('profiles.csv')

### DATA EXTRACTION AND PREPROCESSING

# Additional categories for numerical data

# Number of blank essays for each individual
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
df['blank_essay_count'] = df[essay_cols].isnull().sum(axis=1)

# Replace all NaNs in text fields with empty string
for col in df:
    if col != 'height':
        df[col] = df[col].replace(np.nan,'',regex=True)

# Total Essay Length
all_essays = df[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df['essay_len'] = all_essays.apply(lambda x: len(x))

# Number of words in essay 4 (about favorite books, TV, movies)
df['essay4_len'] = df.essay4.str.split().str.len()

# Number of times "family" mentioned in essay 5, things you couldn't live without
df['essay5_family'] = [df.essay5[i].count('family') for i in range(len(df.essay5))]


        
        
# Map categorical data of interest to numerical code and remove any NaN

# Body Type
body_type_map = {'rather not say': 0, 'used up': 1, 'skinny': 2, 'thin': 3, 'fit': 4, 'average': 5, 'athletic': 6, 'jacked': 7, 'a little extra': 8, 'overweight': 9, 'full figured': 10, 'curvy': 11}
df['body_type_code'] = df['body_type'].map(body_type_map)

# Drug Use
drugs_map = {'never': 0, 'sometimes': 1, 'often': 2}
df['drugs_code'] = df.drugs.map(drugs_map)

# Drinks
drinks_map = {'not at all': 0 , 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4, 'desperately': 5}
df['drinks_code'] = df.drinks.map(drinks_map)

# Sex
sex_map = {'m':0, 'f':1}
df['sex_code'] = df.sex.map(sex_map)

# Main religion classification
df['religion_simple'] = df.religion.str.split().str.get(0)
religion_simple_map = {'other':0, 'christianity':1, 'catholicism':2, 'judaism':3, 'islam':4, 'buddhism':5, 'agnosticism':6, 'atheism':7}
df['religion_simple_code'] = df.religion_simple.map(religion_simple_map)


# Remove any samples (rows) containing NaN and then normalize all data

features_numeric = df[['body_type_code','drugs_code','drinks_code','sex_code','religion_simple_code','essay_len','blank_essay_count','essay4_len','essay5_family','age','height']].dropna()
vals = features_numeric.values
vals_scaled = preprocessing.MinMaxScaler().fit_transform(vals)
features_scaled = pd.DataFrame(vals_scaled, columns = features_numeric.columns)


### EXPLORING DATA

# Any visual relation between length of all essays, length of essay 4 (favorite books, tv shows, movies), 
# and number of times "family" occurs in essay 5 (things you couldn't live without)?

# Possible correlations with intelligence or religiosity?
plt.figure()
plt.plot(features_scaled.blank_essay_count, features_scaled.age, 'b.')
plt.xlabel("# of blank essays")
plt.ylabel("Age")

plt.figure()
plt.plot(features_scaled.essay4_len, features_scaled.age, 'b.')
plt.xlabel("Length of essay 4 (favorite books, tv shows, movies)")
plt.ylabel("Age")

plt.figure()
plt.plot(features_scaled.essay5_family, features_scaled.age, 'b.')
plt.xlabel("# of 'family' mentions in essay 5")
plt.ylabel("Age")

plt.figure()
plt.plot(df.essay5_family, df.religion_simple_code, 'b.', alpha = 0.1)
plt.xlabel("# of 'family' mentions")
plt.ylabel("Religion")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_scaled.essay_len, features_scaled.essay4_len, features_scaled.essay5_family, c='r', marker='o')
ax.set_xlabel("Total essay length")
ax.set_ylabel("Length of essay 4")
ax.set_zlabel("'Family' in essay 5")
plt.tight_layout()


### REGRESSION APPROACHES
# Predicting age with total essay lengths, length of essay 4, number of times "family" occurs in essay 5

# Create training and test sets
x_train = features_scaled[['essay_len','essay4_len','essay5_family']][:-5000]
y_train = features_numeric.age[:-5000]
x_test = features_scaled[['essay_len','essay4_len','essay5_family']][-5000:]
y_test = features_numeric.age[-5000:]

# Multiple Linear Regression
model1 = LinearRegression()
start1 = time.time()
model1.fit(x_train, y_train)
end1 = time.time()
start2 = time.time()
y_predict = model1.predict(x_test)
end2 = time.time()

print("MLR - Time elapsed for fit: {} s".format(end1-start1))
print("MLR - Time elapsed for predict: {} s".format(end2-start2))
print("MLR (validation) R^2")
print(model1.score(x_train,y_train))
print("MLR (prediction) R^2")
print(model1.score(x_test,y_test))           
print("\n")

# K Nearest Neighbors Regression

# Determining best K
scores = []

for i in range(1, 20):
    model2 = KNeighborsRegressor(n_neighbors = i, weights = 'distance')
    model2.fit(x_train,y_train)
    scores.append(model2.score(x_train,y_train))

plt.figure()
plt.plot(list(range(1, 20)), scores)
plt.xlabel('K value')
plt.ylabel('Score')

# Using K = 10
model2 = KNeighborsRegressor(n_neighbors = 10, weights = 'distance')
start1 = time.time()
model2.fit(x_train, y_train)
end1 = time.time()
start2 = time.time()
y_predict = model2.predict(x_test)
end2 = time.time()

print("KNN Regressor - Time elapsed for fit: {} s".format(end1-start1))
print("KNN Regressor - Time elapsed for predict: {} s".format(end2 - start2))
print("KNN Regressor (validation) R^2:")
print(model2.score(x_train,y_train))
print("KNN Regressor (prediction) R^2:")
print(model2.score(x_test,y_test))
print("\n")


### CLASSIFICATION APPROACHES
# Predicting religion from a variety of factors

# Create training and test sets
x_train = features_scaled[['body_type_code','drugs_code','drinks_code','sex_code','essay_len','blank_essay_count','essay4_len','essay5_family','age']][:-5000]
y_train = features_numeric.religion_simple_code[:-5000]
x_test = features_scaled[['body_type_code','drugs_code','drinks_code','sex_code','essay_len','blank_essay_count','essay4_len','essay5_family','age']][-5000:]
y_test = features_numeric.religion_simple_code[-5000:]

# Support Vector Machine
model3 = SVC(kernel='rbf', gamma = 'auto', C=0.01)
start1 = time.time()
model3.fit(x_train,y_train)
end1 = time.time()
start3 = time.time()
y_predict = model3.predict(x_test)
end3 = time.time()

print("SVM - Time elapsed for fit: {} s".format(end1-start1))
print("SVM - Time elapsed for predict: {} s".format(end2 - start2))
print("SVM Accuracy (validation):")
print(accuracy_score(y_train, model3.predict(x_train), normalize=True))
print("SVM Accuracy (test):")
print(accuracy_score(y_test, y_predict))
print("\n")

print("SVM Precision (validation):")
print(precision_score(y_train, model3.predict(x_train), average=None))
print("SVM Precision (test):")
print(precision_score(y_test, y_predict, average=None))
print("\n")

print("SVM Recall (validation):")
print(recall_score(y_train, model3.predict(x_train), average=None))
print("SVM Recall (test):")
print(recall_score(y_test, y_predict, average=None))
print("\n")

# K-Nearest Neighbors Classifier
model4 = KNeighborsClassifier(n_neighbors = 8)
start1 = time.time()
model4.fit(x_train,y_train)
end1 = time.time()
start2 = time.time()
y_predict = model4.predict(x_test)
end2 = time.time()

print("KNN - Time elapsed for fit: {} s".format(end1-start1))
print("KNN - Time elapsed for predict: {} s".format(end2 - start2))
print("KNN Accuracy (validation):")
print(accuracy_score(y_train, model4.predict(x_train), normalize=True))
print("KNN Accuracy (test):")
print(accuracy_score(y_test, y_predict))
print("\n")

print("KNN Precision (validation):")
print(precision_score(y_train, model4.predict(x_train), average=None))
print("KNN Precision (test):")
print(precision_score(y_test, y_predict, average=None))
print("\n")

print("KNN Recall (validation):")
print(recall_score(y_train, model4.predict(x_train), average=None))
print("KNN Recall (test):")
print(recall_score(y_test, y_predict, average=None))
print("\n")



