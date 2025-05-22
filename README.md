# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program and Output:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Shree Lekha.S
RegisterNumber: 212223110052
*/
```
```
import pandas as pd
data = pd.read_csv("Salary.csv")

data.head()
```
![image](https://github.com/user-attachments/assets/09a89a04-611b-4fc3-875d-bd88d6f665d1)

```
data.info()
```
![image](https://github.com/user-attachments/assets/e621a128-89c8-4cad-934f-f376081ab400)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/16b3b57d-07b1-4430-b203-0b6da98cc417)

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/5e54a124-dedb-4ff0-adc0-2e49a572fdc4)

```
x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

```
![image](https://github.com/user-attachments/assets/b7d18527-ef03-497e-b30c-383cca9f5bc5)

```
r2 = metrics.r2_score(y_test, y_pred)
r2

```
![image](https://github.com/user-attachments/assets/da300bb2-0cb9-4af1-80ad-253468836873)
```
dt.predict([[5,6]])
```
![image](https://github.com/user-attachments/assets/ae60c888-db56-4005-8c9b-61f3e56a029c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
