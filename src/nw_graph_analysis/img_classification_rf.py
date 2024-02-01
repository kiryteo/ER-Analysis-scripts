import os
import imageio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Assuming you have a function load_images_and_labels() to load your data
# X should be a 2D array where each row is a flattened image (128x128)
# y should be the corresponding labels
# Adjust this part based on your actual data loading mechanism
X = []
y = []

data = os.listdir('/localhome/asa420/MIAL/data/classification_data/2d_classification/train/')

for file in data:
    frame = imageio.imread('/localhome/asa420/MIAL/data/classification_data/2d_classification/train/' + file)
    X.append(frame.flatten())
    if file[0] == 'a':
        y.append(0)
    elif file[0] == 'c' and file[1] == 'l':
        y.append(1)
    elif file[0] == 'r':
        y.append(3)
    else:
        y.append(2)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Create a Random Forest classifier
# clf = RandomForestClassifier(n_estimators=200, random_state=42)

# # Train the classifier
# clf.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = clf.predict(X_test)

# # Evaluate the performance
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# # Print classification report
# print("Classification Report:")
# print(metrics.classification_report(y_test, y_pred))

# # Print confusion matrix
# print("Confusion Matrix:")
# print(metrics.confusion_matrix(y_test, y_pred))


# from sklearn.neural_network import MLPClassifier

# # Create a simple neural network classifier
# mlp_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
# mlp_clf.fit(X_train, y_train)
# y_pred_mlp = mlp_clf.predict(X_test)

# # Evaluate the performance
# accuracy = metrics.accuracy_score(y_test, y_pred_mlp)
# print(f"Accuracy: {accuracy}")

# # Print classification report
# print("Classification Report:")
# print(metrics.classification_report(y_test, y_pred_mlp))


from xgboost import XGBClassifier

# Create an XGBoost classifier
xgb_clf = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

# Evaluate the performance
accuracy = metrics.accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy}")


# from sklearn.neighbors import KNeighborsClassifier

# # Create a KNN classifier
# knn_clf = KNeighborsClassifier(n_neighbors=5)
# knn_clf.fit(X_train, y_train)
# y_pred_knn = knn_clf.predict(X_test)

# # Evaluate the performance
# accuracy = metrics.accuracy_score(y_test, y_pred_knn)
# print(f"Accuracy: {accuracy}")

# from sklearn.svm import SVC

# # Create an SVM classifier
# svm_clf = SVC(kernel='rbf', C=1)
# svm_clf.fit(X_train, y_train)
# y_pred_svm = svm_clf.predict(X_test)

# # Evaluate the performance
# accuracy = metrics.accuracy_score(y_test, y_pred_svm)
# print(f"Accuracy: {accuracy}")



# import numpy as np
# import imageio
# import matplotlib.pyplot as plt

# img = np.zeros((128, 128))
# for frame in range(1, 32):
#     file = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/Control/er_mean/control{frame}_er_mean.png')
#     img += file

# img = img / 31
# plt.imshow(img)
# plt.show()