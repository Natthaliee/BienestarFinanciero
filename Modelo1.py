#Librer�as utilizadas
#########################################################
import pandas as pnd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib.pylab import rcParams
#rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler


#importación del data set
#########################################################
filename = 'Datos.xlsx' #archivo de datos
data = pnd.read_excel(filename, header = 0) #lee los datos y los guarda en el dataframe

#Muestra datos del data set
#########################################################
print(data.head())  #Imprime los 5 primeras entradas del dataset
print(data.head(5).transpose())  #Imprime los 5 primeras entradas del dataset
print(data.shape) #se muestra la forma, cantidad de entradas y atributos 

#Limpieza del dataset
#########################################################
data.dropna () 
print(data.isna().sum())#El set de datos no contiene valores desconocidos por lo que no es necesario limpiar la data """

#Descripción del dataset/ Caracterización en modo texto 
#########################################################
"""print(data.describe()) #resumen estadístico del dataset
data.info() #información del dataset y tipo de datos para cada columna (data.dtypes) """
data.groupby(['class'])['class'].count()#información por clases
print(data.info())

#Correlación entre variables  y relaciones lineales
#########################################################
#variables numericas
 print(data[['POBLACION', 'MUJERES','HOMBRES', 'CUENTA_M','CUENTA_H','CUENTAS' ,'CREDITOS','PRODUCTOS','POBREZA','PREFERENCIA']].corr())#correlación entre variables

#Distribucion conjunta de las características del set de entrenamiento, CORRELACION.
sns.pairplot(data[["ADULTOS","CUENTAS", "CREDITOS","POBREZA","PRODUCTOS" ]], diag_kind="kde") #creando la gráfica
plt.tight_layout()
plt.show() #muestra la gráfica

#a medida que adquieren un producto financiero, también se adquiere otro
print(data[['CUENTAS', 'CREDITOS']].corr())
sns.regplot(x="CUENTAS", y="CREDITOS", data=data)
plt.show()

#A medida que la población crece, aumenta la adquisición de productos
print(data[['ADULTOS', 'PRODUCTOS']].corr())
sns.regplot(x="ADULTOS", y="PRODUCTOS", data=data)
plt.show()

#A medida que la población crece, aumenta la adquisición de productos
print(data[['PRODUCTOS', 'POBREZA']].corr())
sns.regplot(x="PRODUCTOS", y="POBREZA", data=data)
plt.show()  

#Variables categóricas
#########################################################
#variables categóricas
sns.boxplot(x="REGION", y="PRODUCTOS", data=data)
plt.show()

sns.boxplot(x="REGION", y="POBREZA", data=data)
plt.show()  

#MODELO 
Data_pre = data[['REGION', 'CUENTAS']]
Data_pre.index = Data_pre['REGION']

#plt.plot(Data_pre["CUENTAS"],label='Cuentas por región')
#plt.show()
 
Data_pre = Data_pre.sort_index(ascending=True,axis=0)
data2 = pnd.DataFrame(index=range(0,len(Data_pre)),columns=['REGION','CUENTAS'])
for i in range(0,len(data2)):
    data2["REGION"][i]=Data_pre['REGION'][i]
    data2["CUENTAS"][i]=Data_pre["CUENTAS"][i] 

scaler=MinMaxScaler(feature_range=(0,1))
data2.index=data2.REGION
data2.drop("REGION",axis=1,inplace=True)
final_data = data2.values
train_data=final_data[0:200,:]
valid_data=final_data[200:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_data)
x_train_data,y_train_data=[],[]
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(np.shape(x_train_data)[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
model_data=data2[len(data2)-len(valid_data)-60:].values
model_data=model_data.reshape(-1,1)
model_data=scaler.transform(model_data)

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)
X_test=[]
for i in range(60,model_data.shape[0]):
    X_test.append(model_data[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price=lstm_model.predict(X_test)
predicted_stock_price=scaler.inverse_transform(predicted_stock_price)

train_data=data[:200]
valid_data=data[200:]
valid_data['Predictions']=predicted_stock_price
plt.plot(train_data["CUENTAS"])
plt.plot(valid_data[['CUENTAS',"Predictions"]]) 
 