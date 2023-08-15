#Librer�as utilizadas
#########################################################

import pandas as pnd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#importación del data set
#########################################################
filename = 'Predict.xlsx' #archivo de datos
data = pnd.read_excel(filename, header = 0) #lee los datos y los guarda en el dataframe
data['TIPO'].replace(['Semi-metrópoli','Semi-urbano','Urbano','En Transición','Metrópoli','Sin identificar', 'Rural'],[1,2,3,4,5,6,7], inplace=True)
data = data.fillna(0)
data['TIPO'] = data['TIPO'].astype(int)

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
print(data.describe()) #resumen estadístico del dataset
data.info() #información del dataset y tipo de datos para cada columna (data.dtypes) 

#Creación y ajuste del modelo
#########################################################
prueba = data.select_dtypes(np.number).fillna(0)
objetivo = 'CUENTAS'
independientes = prueba.drop(columns=objetivo).columns

modelo = LinearRegression()
modelo.fit(X=prueba[independientes], y=prueba[objetivo])

#Predicción
data['prediccion'] = modelo.predict(prueba[independientes])
preds = data[["CUENTAS","prediccion"]].head(25)

#prueba del modelo 
resultado = modelo.predict([[32,32058,2050,5]])
print(resultado)

#Gráfica
preds.plot(kind = 'bar', figsize = (18,8))
plt.grid(linewidth='2')
plt.grid(linewidth='2')
plt.grid(None)
plt.show()