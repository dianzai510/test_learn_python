from pyod.utils.data import generate_data
from pyod.utils.example import visualize, data_visualize
X_train, Y_train, X_test, Y_test = generate_data(n_train=5, n_test=2, contamination=0.1)
print(X_train)
data_visualize(X_train=X_train[:,0],y_train= Y_train[:,1])
