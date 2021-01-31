import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
combine = [train_df, test_df]

# Q3
# list numerical data types
# print(train_df._get_numeric_data())

# Q5
# sum per column with null or empty data
# print(train_df.isnull().sum())
print("Null testing data:")
print(test_df.isnull().sum())

# Q7
# list the count, mean, std, min, 25% percentile, 50% percentile,
# 75% percentile, max, of numerical features
print(train_df.Age.describe())
print(train_df.SibSp.describe())
print(train_df.Parch.describe())
print(train_df.Fare.describe())


# Q8
print("Describe categorical features:")
print(train_df.Sex.describe())
print(train_df.Cabin.describe())
print(train_df.Embarked.describe())
print(train_df.Pclass.describe())

# Q9
# correlation Survivability vs pclass
print(train_df.groupby(['Pclass', 'Survived']).count())
pclass_1_df = train_df.query('Pclass == 1')
print(pclass_1_df.head(5))
print(pclass_1_df.corr())

# Q10
# Survivability vs sex
print(train_df.groupby(['Sex', 'Survived']).count())

# Q11
# Histogram between age and Survived
train_age_nosurvive_df = train_df.query('Survived == 0')
train_age_nosurvive_df = train_age_nosurvive_df[['Age']]

train_age_survive_df = train_df.query('Survived == 1')
train_age_survive_df = train_age_survive_df[['Age']]

# train_age_nosurvive_df.plot.hist(bins=30)
# train_age_survive_df.plot.hist(bins=30)


"""
# Q12
# Histogram between Pclass and Survived, with age as the primary variable
train_pclass_1_nosurvive_df = train_df.query('Survived == 0 and Pclass == 1')
train_pclass_1_nosurvive_df = train_pclass_1_nosurvive_df[['Age']]

train_pclass_2_nosurvive_df = train_df.query('Survived == 0 and Pclass == 2')
train_pclass_2_nosurvive_df = train_pclass_2_nosurvive_df[['Age']]

train_pclass_3_nosurvive_df = train_df.query('Survived == 0 and Pclass == 3')
train_pclass_3_nosurvive_df = train_pclass_3_nosurvive_df[['Age']]

# survive = 1
train_pclass_1_survive_df = train_df.query('Survived == 1 and Pclass == 1')
train_pclass_1_survive_df = train_pclass_1_survive_df[['Age']]

train_pclass_2_survive_df = train_df.query('Survived == 1 and Pclass == 2')
train_pclass_2_survive_df = train_pclass_2_survive_df[['Age']]

train_pclass_3_survive_df = train_df.query('Survived == 1 and Pclass == 3')
train_pclass_3_survive_df = train_pclass_3_survive_df[['Age']]
# train_pclass_3_survive_df.plot.hist(bins=30)
# plt.show()

# Checking to see if pclass ages vary
train_pclass_1_df = train_df.query('Pclass == 1')
train_pclass_1_df = train_pclass_1_df[['Age']]

train_pclass_2_df = train_df.query('Pclass == 2')
train_pclass_2_df = train_pclass_2_df[['Age']]

train_pclass_3_df = train_df.query('Pclass == 3')
train_pclass_3_df = train_pclass_3_df[['Age']]

train_pclass_1_df.plot.hist(bins=30)
train_pclass_2_df.plot.hist(bins=30)
train_pclass_3_df.plot.hisprint(train_df.head())t(bins=30)
"""

# Q13
print("Q13 start")
train_survived0_df = train_df.query('Survived == 0')
train_survived0_df = train_survived0_df[['Survived', 'Fare']]
avg_survived0_df = train_survived0_df.groupby(['Survived']).mean()
train_embarkedS_df = train_df.query('Embarked == "S"')
train_embarkedS_df = train_embarkedS_df[['Embarked', 'Fare']]
avg_embarkedS_df = train_embarkedS_df.groupby(['Embarked']).mean()
frames = [avg_survived0_df, avg_embarkedS_df]
plotdata = pd.concat(frames)
plotdata.plot(kind="bar")
plt.show()

# Q14
print("duplicate tickets")
print(train_df.duplicated(subset='Ticket').sum())

# Q15
print("Null cabin values in training data:")
print(train_df['Cabin'].isnull().sum())
print("Null cabin values in testing data:")
print(test_df['Cabin'].isnull().sum())
#train_df.drop(['Cabin'], axis=1)


print(train_df.shape[0])
print(test_df.shape[0])


# Q16
train_df['Sex'] = np.where(train_df['Sex'] == 'male', 0, 1)

# Q17
print('Missing age values:')
print(train_df['Age'].isnull().sum())
# For the KNNImputer to work we need all numeric columns
# Convert Embarked to numeric as well
train_df['Embarked'] = np.where(train_df['Embarked'] == 'C', 0, 0)
train_df['Embarked'] = np.where(train_df['Embarked'] == 'Q', 1, 1)
train_df['Embarked'] = np.where(train_df['Embarked'] == 'S', 2, 2)
# we need to drop any non numberic column
age_complete_df = train_df.drop(['PassengerId', 'Name',
                                 'Ticket', 'Cabin'], axis=1)
imputer = KNNImputer(n_neighbors=5)
age_complete_df = pd.DataFrame(imputer.fit_transform(
    age_complete_df), columns=age_complete_df.columns)
print('Missing age values after replacement:')
print(age_complete_df['Age'].isnull().sum())

# Q18

# Q19
