import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


def plot_decision_boundary(x_x, m_model):
    x_span = np.linspace(min(x_x[:, 0]) - 1, max(x_x[:, 0]) + 1)
    y_span = np.linspace(min(x_x[:, 1]) - 1, max(x_x[:, 1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    predictions = np.argmax(m_model.predict(grid), axis=-1)
    z = predictions.reshape(xx.shape)
    plt.contourf(xx, yy, z)


n_pts = 500

centers_coordinates = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]]
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers_coordinates, cluster_std=0.3)

# Hot encoding to eliminate any dependent relationship between the labels. Hot encoding is necessary when classifying
# models with multiple classes, because it converts a flat list of labels into a table of labels.
y_categorical = to_categorical(y=y, num_classes=5)

model = Sequential()
model.add(Dense(units=5, input_shape=(2,), activation='softmax'))
model.compile(Adam(0.1), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X, y=y_categorical, verbose=1, batch_size=50, epochs=100)

plot_decision_boundary(x_x=X, m_model=model)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.scatter(X[y == 3, 0], X[y == 3, 1])
plt.scatter(X[y == 4, 0], X[y == 4, 1])

# x_point = 0.5
# y_point = -1
# point = np.array([[x_point, y_point]])
# point_prediction = np.argmax(model.predict(point), axis=-1)
# plt.plot([x_point], [y_point], marker='o', markersize=10, color='pink')

# print("Prediction for the point is: ", point_prediction)

plt.show()
