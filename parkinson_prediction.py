import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

parkinsons_data = pd.read_csv("parkinsons.csv")

## Independent and Dependent features
x = parkinsons_data.drop(columns = ['name', 'status'], axis =1)
y = parkinsons_data['status']

##Train Test Split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=.2, random_state=2)

## Standardize the dataset
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

pickle.dump(scaler,open('scaling.pkl','wb'))


model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

### Prediction With Test Data
x_train_prediction = model.predict(x_train)


##transformation of new data
input_data = (119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)
input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshape)



pickle.dump(model,open('svmodel.pkl' , 'wb'))
pickled_model=pickle.load(open('svmodel.pkl','rb'))