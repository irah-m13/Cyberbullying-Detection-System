import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D,Input,Reshape,BatchNormalization, MaxPool1D,Conv1D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow import keras
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
#%% one-hot-encoding
y_train = keras.utils.to_categorical(y_train,2)
y_test  = keras.utils.to_categorical(y_test,2)
inputs = keras.layers.Input(shape=(X_train.shape[1],1))
RS0    = keras.layers.Reshape((X_train.shape[1], ))(inputs)
FC0    = keras.layers.Dense(512, bias_initializer=keras.initializers.VarianceScaling())(RS0)
BN0    = keras.layers.BatchNormalization(axis=-1)(FC0)
AC0    = keras.layers.Activation('relu')(BN0)
DP0    = keras.layers.Dropout(0.2)(AC0)

FC1    = keras.layers.Dense(128, bias_initializer=keras.initializers.VarianceScaling())(DP0)
BN1    = keras.layers.BatchNormalization(axis=-1)(FC1)
AC1    = keras.layers.Activation('relu')(BN1)
DP1    = keras.layers.Dropout(0.2)(AC1)


FC2 =   keras.layers.Dense(2, bias_initializer=keras.initializers.VarianceScaling())(DP1)
outputs = keras.layers.Activation('softmax')(FC2)

myMLP = keras.Model(inputs=inputs,outputs=outputs)
myMLP.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
myMLP.summary()

np.where(y_train==0)[0].shape
np.where(y_train==1)[0].shape
class_weight = {0: 1, 1: 40}

myMLP.fit(X_train,y_train,epochs=30,batch_size=1200,verbose=1, class_weight=class_weight)
filename = 'cnn.sav'
pickle.dump(myMLP, open(filename, 'wb'))
loss_and_metrics = myMLP.evaluate(X_test,y_test)
i=(10-float(loss_and_metrics[1]))*10
print("Accuracy : ",i)

model_dir = "./cnn_model"

localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
myMLP.save(model_dir, options=localhost_save_option)