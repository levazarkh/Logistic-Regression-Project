import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# reading csv file and set it to a DataFrame
ad_data = pd.read_csv('advertising.csv')
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())

# exploratory data analysis

sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
plt.show()

# Area Income versus Age
sns.jointplot(x='Age', y='Area Income', data=ad_data)
plt.show()

sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, color='red', kind='kde')
plt.show()

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')
plt.show()

sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
plt.show()

# Splitting data into training set and testing set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Predictions
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
print(classification_report(y_test,predictions))
