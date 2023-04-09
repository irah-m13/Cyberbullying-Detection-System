import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
data = pd.read_csv('data.csv')
data.head()
data.shape
X = data.iloc[:,:-1]
X.head()
y = data.iloc[:,-1]
y.head()
data['Target'].value_counts()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
sns.countplot(x='Target',data=data)
plt.show()
X_train.shape
X_train.head()
y_test.shape
y_test.head()



from sklearn.svm import SVC
model = SVC(kernel = 'linear')
model.fit(X_train,y_train)
filename = 'svm.sav'
pickle.dump(model, open(filename, 'wb'))
y_pred_rf = model.predict(X_test)
acc=(metrics.accuracy_score(y_pred_rf,y_test)*100)
print("Accuracy is:",acc)
j=acc


cm1 = metrics.confusion_matrix(y_pred_rf,y_test)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)


import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from sklearn.metrics import classification_report

print('\nClassification Report\n')
print(classification_report(y_test,y_pred_rf))




confusion_mtx = confusion_matrix(y_test,y_pred_rf) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
