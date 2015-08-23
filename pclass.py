__author__ = 'Tavleen'
import numpy as np
import pandas as pd
from collections import defaultdict
import pylab as plt

#read train and test files with pandas
train = pd.read_csv(r'C:\Users\HP1\PycharmProjects\titanic\input\train.csv', dtype={"Age": np.float64},)
test = pd.read_csv(r'C:\Users\HP1\PycharmProjects\titanic\input\test.csv', dtype={"Age": np.float64},)

#see the statistics summary of the data
print("\nSummary statistics of training data")
print(train.describe())

total=len(train[train['Survived']])
num_survived=len(train[train['Survived']==1])
percent_survived=float(num_survived/total)*100

unique_pclass=list(set(train['Pclass']))
print(unique_pclass)
#assuming that pclass=1 would survive more
number_surv=defaultdict(float)
number_dead=defaultdict(float)
total_class=defaultdict(float)
for clas in unique_pclass:
    total_class[clas]=len(train[train['Pclass']==clas])
    number_surv[clas]=float(np.sum(train[train['Pclass']==clas]['Survived']==1))
    number_dead[clas]=total_class[clas]-number_surv[clas]

#percentage of people survived from each class
for clas in unique_pclass:
 print("Percentage of survivors from pclass %d is %f" %(int(clas) ,float(number_surv[clas]*100/total_class[clas])))
#print("Percentage of survivors from pclass=2 is %f" %float(number_surv[2]*100/total_class[2]))
#print("Percentage of survivors from pclass=1 is %f" %float(number_surv[3]*100/total_class[3]))
pclass1_surv=pd.Series((number_surv[1],number_dead[1]),name='')
pclass1_chart=pclass1_surv.plot(kind='pie',title='Pclass1 survivors',labels=['Survived','Dead'],figsize=(4,4))
pclass1_chart.get_figure().savefig('Pclass1_survival.png')
plt.clf()

pclass2_surv=pd.Series((number_surv[2],number_dead[2]),name='')
pclass2_chart=pclass2_surv.plot(kind='pie',title='Pclass2 survivors',labels=['Survived','Dead'],figsize=(4,4))
pclass2_chart.get_figure().savefig('Pclass2_survival.png')
plt.clf()

pclass3_surv=pd.Series((number_surv[3],number_dead[3]),name='')
pclass3_chart=pclass3_surv.plot(kind='pie',title='Pclass3 survivors',labels=['Survived','Dead'],figsize=(4,4))
pclass3_chart.get_figure().savefig('Pclass3_survival.png')
plt.clf()

#mean_pclass1_pclass3_surv=(float(number_surv[1]*100/total_class[1])+float(number_surv[3]*100/total_class[3]))/2
#sprint(mean_pclass1_pclass3_surv)

predict=pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})
predict['Survived']=[1 if x==1 else 0 for x in test['Pclass']]
predict.to_csv('pclass.csv',index=False)




