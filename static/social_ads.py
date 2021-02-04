import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('Social_Network_Ads.csv')
#print(df.head(10))

x = df[['Age','EstimatedSalary']]
y = df['Purchased']

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.3)
#k nearest neighbor classification model
'''model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print(acc)'''

#support vector machine model
'''best = 0
for _ in range(30):
	model = svm.SVC(kernel='linear',C=3)
	model.fit(x_train,y_train)

	acc = model.score(x_test,y_test)
	print(acc)

	if acc>best:
		best = acc
		with open('ads.pickle','wb') as file:
			pickle.dump(model,file)'''
			
#random forest classifier model
'''best=0
for _ in range(30):
	x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.3)
	model = RandomForestClassifier(n_estimators=100)

	model.fit(x_train,y_train)
	predict = model.predict(x_test)
	acc = sklearn.metrics.accuracy_score(y_test,predict)
	print(acc)
	if acc>best:
		best = acc
		with open('social_ads.pickle','wb') as file:
			pickle.dump(model,file)'''
file = open('social_ads.pickle','rb')
model = pickle.load(file)

prediction = model.predict(x_test)
