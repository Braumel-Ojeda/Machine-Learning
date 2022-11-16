import encodings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('heart.csv', sep=',', engine='python')

x = df.drop('target',axis=1).values    					#corpus sin etiquetas 
y = df['target'].values 									#etiquetas

print('\n df', df)	
print('\n Corpus')
print('\n', *x)
print ('----------------------')
print('\n Etiquetas')
print('\n', *y)
print ('----------------------')	            

x_e, x_p, y_e, y_p = train_test_split(x, y, test_size=0.4, shuffle = False)	

print('\n Conjunto de entrenamiento')		
print ('\n x_e ', *x_e)
print ('\n y_e ', y_e)
print ('----------------------')


print('\n Conjunto de prueba')	
print ('\n x_p', *x_p)
print ('\n y_p', y_p)
print ('----------------------')

np.savetxt("x_e.csv", x_e, delimiter=",", fmt="%d", header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", comments='')
np.savetxt("y_e.csv", y_e, delimiter=",", fmt="%d", header="target", comments='')
np.savetxt("x_p.csv", x_p, delimiter=",", fmt="%d", header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", comments='')
np.savetxt("y_p.csv", y_p, delimiter=",", fmt="%d", header="target", comments='')
    
