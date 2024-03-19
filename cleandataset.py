import pandas as pd



url = "C:/Users/iFix/Desktop/machine/Diabetes.csv"
dataset = pd.read_csv(url)


columns_to_remove_outliers = ["NPG", "PGL", "DIA", "TSF", "INS", "BMI", "DPF", "AGE", "Diabetic"]


def remove_outliers(dataset, column):
    q1 = dataset[column].quantile(0.25)
    q3 = dataset[column].quantile(0.75)
    iqr = q3 - q1

    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)

    new_df = dataset.loc[(dataset[column] < upper_limit) & (dataset[column] > lower_limit)]
    return new_df



clean_data = dataset.copy()

for column in columns_to_remove_outliers:
    clean_data = remove_outliers(clean_data, column)



print('Before removing outliers: ', len(dataset))
print('After removing outliers: ', len(clean_data))
print('Outliers removed: ', len(dataset) - len(clean_data))





