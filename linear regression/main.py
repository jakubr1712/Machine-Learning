import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("./honeyproduction.csv")


print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = prod_per_year.year

X = X.values.reshape(-1, 1)
print(X)

y = prod_per_year.totalprod
print(y)


regr = linear_model.LinearRegression()

regr.fit(X, y)

print(regr.coef_, regr.intercept_)

y_predict = regr.predict(X)


X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)
print(X_future)

plt.scatter(X, y)
plt.plot(X, y_predict)
plt.show()
plt.plot(X_future, future_predict)

plt.show()
