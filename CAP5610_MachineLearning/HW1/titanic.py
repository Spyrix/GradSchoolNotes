import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
combine = [train_df, test_df]

# Q1
####
# No code necessary

# Q2
####
# No code necessary

# Q3
####
# list numerical data types
print(train_df._get_numeric_data())

# Q5
####
# sum per column with null or empty data
print("Q5:")
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Q6
####
# No code necessary

# Q7
####
# list the count, mean, std, min, 25% percentile, 50% percentile,
# 75% percentile, max, of numerical features
print("Q7:")
print(train_df.Age.describe())
print(train_df.SibSp.describe())
print(train_df.Parch.describe())
print(train_df.Fare.describe())
print(train_df.Pclass.describe())
print(train_df.Survived.describe())

# Q8
####
print("Q8: Describe categorical features:")
print(train_df.Sex.describe())
print(train_df.Cabin.describe())
print(train_df.Embarked.describe())

# Q9
####
# correlation Survivability vs pclass
print("Q9:")
print(train_df.groupby(['Pclass', 'Survived']).count())

# Q10
#####
# Survivability vs sex
"Q10: "
print(train_df.groupby(['Sex', 'Survived']).count())

# Q11
#####
# Histogram between age and Survived
train_age_nosurvive_df = train_df.query('Survived == 0')
train_age_nosurvive_df = train_age_nosurvive_df[['Age']]

train_age_survive_df = train_df.query('Survived == 1')
train_age_survive_df = train_age_survive_df[['Age']]

# train_age_nosurvive_df.plot.hist(bins=30)
# train_age_survive_df.plot.hist(bins=30)


# Q12
#####

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

# train_pclass_1_df.plot.hist(bins=30)
# train_pclass_2_df.plot.hist(bins=30)
# train_pclass_3_df.plot.hisprint(train_df.head())t(bins=30)


# Q13
#####
print("Q13:")
# Survived = 0 and Embarked from S
train_survived0_embarkedS_df = train_df.query('Survived == 0 and Embarked=="S"')
train_survived0_embarkedS_df = train_survived0_embarkedS_df[['Sex', 'Fare']]
avg_survived0_embarkedS_df = train_survived0_embarkedS_df.groupby(['Sex']).mean()

frames_S_0 = [avg_survived0_embarkedS_df.query(
    'Sex=="male"'), avg_survived0_embarkedS_df.query('Sex=="female"')]
plotdata_S_0 = pd.concat(frames_S_0)

# Survived = 1 and Embarked from S
train_survived1_embarkedS_df = train_df.query('Survived == 1 and Embarked=="S"')
train_survived1_embarkedS_df = train_survived1_embarkedS_df[['Sex', 'Fare']]
avg_survived1_embarkedS_df = train_survived1_embarkedS_df.groupby(['Sex']).mean()

frames_S_1 = [avg_survived1_embarkedS_df.query(
    'Sex=="male"'), avg_survived1_embarkedS_df.query('Sex=="female"')]
plotdata_S_1 = pd.concat(frames_S_1)

# Survived = 0 and Embarked from C
train_survived0_embarkedC_df = train_df.query('Survived == 0 and Embarked=="C"')
train_survived0_embarkedC_df = train_survived0_embarkedC_df[['Sex', 'Fare']]
avg_survived0_embarkedC_df = train_survived0_embarkedC_df.groupby(['Sex']).mean()

frames_C_0 = [avg_survived0_embarkedC_df.query(
    'Sex=="male"'), avg_survived0_embarkedC_df.query('Sex=="female"')]
plotdata_C_0 = pd.concat(frames_C_0)

# Survived = 1 and Embarked from C
train_survived1_embarkedC_df = train_df.query('Survived == 1 and Embarked=="C"')
train_survived1_embarkedC_df = train_survived1_embarkedC_df[['Sex', 'Fare']]
avg_survived1_embarkedC_df = train_survived1_embarkedC_df.groupby(['Sex']).mean()

frames_C_1 = [avg_survived1_embarkedC_df.query(
    'Sex=="male"'), avg_survived1_embarkedC_df.query('Sex=="female"')]
plotdata_C_1 = pd.concat(frames_C_1)

# Survived = 0 and Embarked from Q
train_survived0_embarkedQ_df = train_df.query('Survived == 0 and Embarked=="Q"')
train_survived0_embarkedQ_df = train_survived0_embarkedQ_df[['Sex', 'Fare']]
avg_survived0_embarkedQ_df = train_survived0_embarkedQ_df.groupby(['Sex']).mean()

frames_Q_0 = [avg_survived0_embarkedQ_df.query(
    'Sex=="male"'), avg_survived0_embarkedQ_df.query('Sex=="female"')]
plotdata_Q_0 = pd.concat(frames_Q_0)

# Survived = 1 and Embarked from Q
train_survived1_embarkedQ_df = train_df.query('Survived == 1 and Embarked=="Q"')
train_survived1_embarkedQ_df = train_survived1_embarkedQ_df[['Sex', 'Fare']]
avg_survived1_embarkedQ_df = train_survived1_embarkedQ_df.groupby(['Sex']).mean()

frames_Q_1 = [avg_survived1_embarkedQ_df.query(
    'Sex=="male"'), avg_survived1_embarkedQ_df.query('Sex=="female"')]
plotdata_Q_1 = pd.concat(frames_Q_1)

# plotdata_Q_1.plot(kind="bar")
# plt.show()

# Q14
#####
print("Q14:")
print(train_df.duplicated(subset='Ticket').sum())

# Q15
#####
print("Q15:")
print("Null cabin values in training data:")
print(train_df['Cabin'].isnull().sum())
print("Null cabin values in testing data:")
print(test_df['Cabin'].isnull().sum())
#train_df.drop(['Cabin'], axis=1)
print("number of training rows:")
print(train_df.shape[0])
#train_df.drop(['Cabin'], axis=1)
print("number of test rows:")
print(test_df.shape[0])


# Q16
#####
train_df['Sex'] = np.where(train_df['Sex'] == 'male', 0, 1)
train_df.rename(columns={"Sex": "Gender"}, inplace=True)
train_df.astype({'Gender': 'int32'})

# Q17
#####
print('Q17:')
print('Missing age values:')
print(train_df['Age'].isnull().sum())
# For the KNNImputer to work we need all numeric columns
age_complete_df = train_df.copy(deep=True)
# we need to drop any non numberic column
age_complete_df = age_complete_df.drop(['PassengerId', 'Name',
                                        'Ticket', 'Cabin'], axis=1)
# Convert Embarked to numeric as well
age_complete_df['Embarked'].replace('C', 0, inplace=True)
age_complete_df['Embarked'].replace('Q', 1, inplace=True)
age_complete_df['Embarked'].replace('S', 2, inplace=True)

# Use the imputter
imputer = KNNImputer(n_neighbors=5)
age_complete_df = pd.DataFrame(imputer.fit_transform(
    age_complete_df), columns=age_complete_df.columns)
print('Missing age values after replacement:')
print(age_complete_df['Age'].isnull().sum())

# Q18
#####
print('Q18:')
print('Missing Embarked values:')
print(train_df.Embarked.isnull().sum())
print(train_df.Embarked.describe())
# according to describe, 'S' is the most common value in Embarked
train_df.Embarked.fillna('S', inplace=True)
print('Missing Embarked values:')
print(train_df.Embarked.isnull().sum())

# Q19
#####
print('Q19:')
print('Missing Test Fare values:')
print(test_df.Fare.isnull().sum())
test_df.Fare.fillna(test_df.Fare.mode()[0], inplace=True)
print('Missing Test Fare values:')
print(test_df.Fare.isnull().sum())

# Q20
#####
print('Q20:')
bins = [-0.001, 7.91, 14.454, 31.0, 512.329]
labels = [0, 1, 2, 3]
train_df['Fare Ordinal Label'] = pd.cut(train_df['Fare'], bins, labels=labels)
print(train_df[['Fare', 'Fare Ordinal Label']].head(5))
