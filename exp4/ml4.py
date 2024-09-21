import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics   
car_data = pd.read_csv('C:\\Users\\boyel\\Desktop\\ml lab\\exp4\\cars dataset.csv') 
car_data.head() 
car_data.info()  
car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True) 
car_data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True) 
car_data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True) 
corrMatrix = car_data.corr() 
sns.heatmap(corrMatrix, annot=True, cmap="viridis") 
plt.show() 
X = car_data.drop(['car_ID','price'],axis=1) 
Y = car_data['price'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42) 
lin_reg_model = LinearRegression() 
lin_reg_model.fit(X_train,Y_train) 
training_data_prediction = lin_reg_model.predict(X_train) 
train_error_score = metrics.r2_score(Y_train, training_data_prediction) 
print("R squared Error - Training : ", train_error_score)  
Y_pred = lin_reg_model.predict(X_test) 
test_error_score = metrics.r2_score(Y_test, Y_pred) 
print("R squared Error - Test: ", test_error_score)  
sns.regplot(Y_test,Y_pred,scatter_kws={"color": "green"},line_kws={"color": "blue"})
