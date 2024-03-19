import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from cleandataset import clean_data




# Task 1: Summary statistics of all attributes
pd.set_option('display.max_columns', None)
summary_statistics = clean_data.describe()
print(summary_statistics)

# Task 2: Distribution of the class label (Diabetic)
plt.figure(figsize=(6, 4))
sns.countplot(x='Diabetic', data=clean_data)
plt.title('Distribution of Diabetic Class')
plt.show()

#Task 3: Create histograms for diabetics in different age groups
plt.figure(figsize=(16, 8))
sns.histplot(x='AGE', hue='Diabetic', data=clean_data, bins=6, kde=True, multiple="stack", palette="husl")
plt.title('Distribution of Diabetics in Different Age Groups')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Diabetic', labels=['Non-Diabetic', 'Diabetic'])
plt.show()

#Task 4: Show the density plot for age
plt.figure(figsize=(10, 6))
sns.distplot(clean_data['AGE'],  color='green')
plt.title('Density Plot for Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

#Task 5: Show the density plot for BMI
plt.figure(figsize=(10, 6))
sns.distplot(clean_data['BMI'], hist=False, color='green')
plt.title('Density Plot for BMI')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.show()

#Task 6: correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = clean_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Features')
plt.show()

#Task 7: split the data into 80% training 20% testing

X = clean_data.drop('Diabetic', axis=1)
y = clean_data['Diabetic']

# Split the dataset into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)







