__author__ = 'Tavleen'
import numpy as np
import pandas as pd
from collections import defaultdict
import pylab as plt

#read train and test files with pandas
train = pd.read_csv(r'C:\Users\HP1\PycharmProjects\titanic\input\train.csv', dtype={"Age": np.float64},)
test = pd.read_csv(r'C:\Users\HP1\PycharmProjects\titanic\input\test.csv', dtype={"Age": np.float64},)

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head(2))
print("\n\nSummary statistics of training data")
print(train.describe())
count =0
#Number of missing or NA in age

df_age=train["Age"]
print(df_age.shape[0])
print("Number of Missing Values in Age is %d" %(df_age.shape[0]-df_age.dropna().shape[0]))


#Calculate percentage of people who survived
num_survived=len(train[train['Survived']==1])

total=len(train[['Survived']])
percent_survived=float(num_survived)/total
print("Percentage of people who survived is %f" %(percent_survived*100))

#Percentage of male and female survivors
unique_sex=pd.unique(train['Sex'].ravel())
number_survived=defaultdict(float)
number_dead=defaultdict(float)
total_sex=defaultdict(float)
print(unique_sex)
for sex in unique_sex:
	total_sex[sex]=len(train[train['Sex']==sex])

	number_survived[sex]=float(np.sum(train[train['Sex']==sex]['Survived']==1))
	number_dead[sex]=total_sex[sex]-number_survived[sex]


print("Percentage of female Survivors is %f" %(float(number_survived['female']*100)/float(total_sex['female'])))
print("Percentage of male Survivors is %f" %(float(number_survived['male']*100)/float(total_sex['male'])))

#For Gender based Model, let us predict, Survival=1 if female else Survival=0
female_survival=pd.Series((number_survived['female'],number_dead['female']),name='')
female_chart=female_survival.plot(kind='pie',title='Female Survival',labels=['Survived','Died'],colors=['#21BFFF','#90DFFF'],figsize=(4,4), autopct='%1.1f%%')
female_chart.set_aspect('equal')
fig_female = female_chart.get_figure()
fig_female.savefig('x1_female_survival.png')
plt.clf()

male_survival=pd.Series((number_survived['male'],number_dead['male']),name='')
male_chart=male_survival.plot(kind='pie',title='Male Survival',labels=['Survived','Died'],figsize=(4,4))
fig_male=male_chart.get_figure()
fig_male.savefig('x1_male_survival.png')
plt.clf()

predict=pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})
predict['Survived']=[1 if x=='female' else 0 for x in test['Sex']]
predict.to_csv("genderbasedmodel.csv",index=False)
#if predict['Survived'] == train['Survived']:
 #   count = count + 1
#accuracy = count/len(train)
accuracy = sum(predict['Survived'] == train['Survived'])/len(predict['Survived'])
print accuracy
