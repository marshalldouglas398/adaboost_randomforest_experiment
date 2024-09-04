# Imports
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

# All of the Iris Values to Average Performance Metrics
# Iris AdaBoost
iris_ada_time = []
iris_ada_balanced_accuracy = []
iris_ada_accuracy = []
iris_ada_recall = []
iris_ada_f1 = []
iris_ada_precision = []
iris_ada_jaccard = []

# Iris Random Forest
iris_rf_time = []
iris_rf_balanced_accuracy = []
iris_rf_accuracy = []
iris_rf_recall = []
iris_rf_f1 = []
iris_rf_precision = []
iris_rf_jaccard = []

# Train the Iris AdaBoost Model 100 Times
for i in range(0, 100):
    # Build the Iris Dataset
    print("Iris dataset working. . . ")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    print("Data split successfully!")

    # Create the AdaBoost Classifier
    ada = AdaBoostClassifier(n_estimators = 100, random_state = 42)

    # Collect Time Statistic
    ada_start = time.time()

    # Fit the AdaBoost Model
    ada.fit(X_train, y_train)
    ada_stop = time.time()
    ada_time = ada_stop - ada_start
    iris_ada_time.append(ada_time)
    print("AdaBoost Classifier trained!")

    # Build the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

    # Collect Time Statistic
    rf_start = time.time()
    rf.fit(X_train, y_train)
    rf_stop = time.time()
    rf_time = rf_stop - rf_start
    iris_rf_time.append(rf_time)
    print("Random Forest Classifier trained!")

    # Calculate All of the Metrics
    balanced_accuracy_ada = balanced_accuracy_score(ada.predict(X_test), y_test)
    iris_ada_balanced_accuracy.append(balanced_accuracy_ada)
    accuracy_ada = accuracy_score(ada.predict(X_test), y_test)
    iris_ada_accuracy.append(accuracy_ada)
    recall_ada = recall_score(ada.predict(X_test), y_test, average = 'macro')
    iris_ada_recall.append(recall_ada)
    precision_ada = precision_score(ada.predict(X_test), y_test, average = "macro")
    iris_ada_precision.append(precision_ada)
    f1_ada = f1_score(ada.predict(X_test), y_test, average = "macro")
    iris_ada_f1.append(f1_ada)
    jaccard_ada = jaccard_score(ada.predict(X_test), y_test, average = "macro")
    iris_ada_jaccard.append(jaccard_ada)
    balanced_accuracy_rf = balanced_accuracy_score(rf.predict(X_test), y_test)
    iris_rf_balanced_accuracy.append(balanced_accuracy_rf)
    accuracy_rf = accuracy_score(rf.predict(X_test), y_test)
    iris_rf_accuracy.append(accuracy_rf)
    recall_rf = recall_score(rf.predict(X_test), y_test, average = 'macro')
    iris_rf_recall.append(recall_rf)
    precision_rf = precision_score(rf.predict(X_test), y_test, average = "macro")
    iris_rf_precision.append(precision_rf)
    f1_rf = f1_score(rf.predict(X_test), y_test, average = "macro")
    iris_rf_f1.append(f1_rf)
    jaccard_rf = jaccard_score(rf.predict(X_test), y_test, average = "macro")
    iris_rf_jaccard.append(jaccard_rf)

wine_ada_time = []
wine_ada_balanced_accuracy = []
wine_ada_accuracy = []
wine_ada_recall = []
wine_ada_f1 = []
wine_ada_precision = []
wine_ada_jaccard = []

wine_rf_time = []
wine_rf_balanced_accuracy = []
wine_rf_accuracy = []
wine_rf_recall = []
wine_rf_f1 = []
wine_rf_precision = []
wine_rf_jaccard = []

for i in range(0, 100):
    print("Wine dataset working. . . ")
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    print("Data split successfully!")

    ada = AdaBoostClassifier(n_estimators = 100, random_state = 42)
    ada_start = time.time()
    ada.fit(X_train, y_train)
    ada_stop = time.time()
    ada_time = ada_stop - ada_start
    wine_ada_time.append(ada_time)
    print("AdaBoost Classifier trained!")

    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
    rf_start = time.time()
    rf.fit(X_train, y_train)
    rf_stop = time.time()
    rf_time = rf_stop - rf_start
    wine_rf_time.append(rf_time)
    print("Random Forest Classifier trained!")

    balanced_accuracy_ada = balanced_accuracy_score(ada.predict(X_test), y_test)
    wine_ada_balanced_accuracy.append(balanced_accuracy_ada)
    accuracy_ada = accuracy_score(ada.predict(X_test), y_test)
    wine_ada_accuracy.append(accuracy_ada)
    recall_ada = recall_score(ada.predict(X_test), y_test, average = 'macro')
    wine_ada_recall.append(recall_ada)
    precision_ada = precision_score(ada.predict(X_test), y_test, average = "macro")
    wine_ada_precision.append(precision_ada)
    f1_ada = f1_score(ada.predict(X_test), y_test, average = "macro")
    wine_ada_f1.append(f1_ada)
    jaccard_ada = jaccard_score(ada.predict(X_test), y_test, average = "macro")
    wine_ada_jaccard.append(jaccard_ada)
    balanced_accuracy_rf = balanced_accuracy_score(rf.predict(X_test), y_test)
    wine_rf_balanced_accuracy.append(balanced_accuracy_rf)
    accuracy_rf = accuracy_score(rf.predict(X_test), y_test)
    wine_rf_accuracy.append(accuracy_rf)
    recall_rf = recall_score(rf.predict(X_test), y_test, average = 'macro')
    wine_rf_recall.append(recall_rf)
    precision_rf = precision_score(rf.predict(X_test), y_test, average = "macro")
    wine_rf_precision.append(precision_rf)
    f1_rf = f1_score(rf.predict(X_test), y_test, average = "macro")
    wine_rf_f1.append(f1_rf)
    jaccard_rf = jaccard_score(rf.predict(X_test), y_test, average = "macro")
    wine_rf_jaccard.append(jaccard_rf)

breast_cancer_ada_time = []
breast_cancer_ada_balanced_accuracy = []
breast_cancer_ada_accuracy = []
breast_cancer_ada_recall = []
breast_cancer_ada_f1 = []
breast_cancer_ada_precision = []
breast_cancer_ada_jaccard = []

breast_cancer_rf_time = []
breast_cancer_rf_balanced_accuracy = []
breast_cancer_rf_accuracy = []
breast_cancer_rf_recall = []
breast_cancer_rf_f1 = []
breast_cancer_rf_precision = []
breast_cancer_rf_jaccard = []

for i in range(0, 100):
    print(f"Breast Cancer dataset working {i}. . . ")
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    print("Data split successfully!")

    ada = AdaBoostClassifier(n_estimators = 100, random_state = 42)
    ada_start = time.time()
    ada.fit(X_train, y_train)
    ada_stop = time.time()
    ada_time = ada_stop - ada_start
    breast_cancer_ada_time.append(ada_time)
    print("AdaBoost Classifier trained!")

    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
    rf_start = time.time()
    rf.fit(X_train, y_train)
    rf_stop = time.time()
    rf_time = rf_stop - rf_start
    breast_cancer_rf_time.append(rf_time)
    print("Random Forest Classifier trained!")

    balanced_accuracy_ada = balanced_accuracy_score(ada.predict(X_test), y_test)
    breast_cancer_ada_balanced_accuracy.append(balanced_accuracy_ada)
    accuracy_ada = accuracy_score(ada.predict(X_test), y_test)
    breast_cancer_ada_accuracy.append(accuracy_ada)
    recall_ada = recall_score(ada.predict(X_test), y_test, average = 'macro')
    breast_cancer_ada_recall.append(recall_ada)
    precision_ada = precision_score(ada.predict(X_test), y_test, average = "macro")
    breast_cancer_ada_precision.append(precision_ada)
    f1_ada = f1_score(ada.predict(X_test), y_test, average = "macro")
    breast_cancer_ada_f1.append(f1_ada)
    jaccard_ada = jaccard_score(ada.predict(X_test), y_test, average = "macro")
    breast_cancer_ada_jaccard.append(jaccard_ada)
    balanced_accuracy_rf = balanced_accuracy_score(rf.predict(X_test), y_test)
    breast_cancer_rf_balanced_accuracy.append(balanced_accuracy_rf)
    accuracy_rf = accuracy_score(rf.predict(X_test), y_test)
    breast_cancer_rf_accuracy.append(accuracy_rf)
    recall_rf = recall_score(rf.predict(X_test), y_test, average = 'macro')
    breast_cancer_rf_recall.append(recall_rf)
    precision_rf = precision_score(rf.predict(X_test), y_test, average = "macro")
    breast_cancer_rf_precision.append(precision_rf)
    f1_rf = f1_score(rf.predict(X_test), y_test, average = "macro")
    breast_cancer_rf_f1.append(f1_rf)
    jaccard_rf = jaccard_score(rf.predict(X_test), y_test, average = "macro")
    breast_cancer_rf_jaccard.append(jaccard_rf)

X = ['AdaBoost', 'RandomForest']
iris_time = [np.mean(iris_ada_time), np.mean(iris_rf_time)]
iris_balanced_accuracy = [np.mean(iris_ada_balanced_accuracy), np.mean(iris_rf_balanced_accuracy)]
iris_accuracy = [np.mean(iris_ada_accuracy), np.mean(iris_rf_accuracy)]
iris_recall = [np.mean(iris_ada_recall), np.mean(iris_rf_recall)]
iris_precision = [np.mean(iris_ada_precision), np.mean(iris_rf_precision)]
iris_f1 = [np.mean(iris_ada_f1), np.mean(iris_rf_f1)]
iris_jaccard = [np.mean(iris_ada_jaccard), np.mean(iris_rf_jaccard)]
iris_time_err = [np.std(iris_ada_time), np.std(iris_rf_time)]
iris_balanced_accuracy_err = [np.std(iris_ada_balanced_accuracy), np.std(iris_rf_balanced_accuracy)]
iris_accuracy_err = [np.std(iris_ada_accuracy), np.std(iris_rf_accuracy)]
iris_recall_err = [np.std(iris_ada_recall), np.std(iris_rf_recall)]
iris_precision_err = [np.std(iris_ada_precision), np.std(iris_rf_precision)]
iris_f1_err = [np.std(iris_ada_f1), np.std(iris_rf_f1)]
iris_jaccard_err = [np.std(iris_ada_jaccard), np.std(iris_rf_jaccard)]

n = 2
r = np.arange(n)
width = 0.1

plt.bar(r, iris_time, width = width, edgecolor = "black", label = "Time")
plt.bar(r + width, iris_balanced_accuracy, width = width, edgecolor = "black", label = "Balanced Accuracy")
plt.bar(r + 2 * width, iris_accuracy, width = width, edgecolor = "black", label = "Accuracy")
plt.bar(r + 3 * width, iris_recall, width = width, edgecolor = "black", label = "Recall")
plt.bar(r + 4 * width, iris_precision, width = width, edgecolor = "black", label = "Precision")
plt.bar(r + 5 * width, iris_f1, width = width, edgecolor = "black", label = "F1-Score")

plt.xlabel("Algorithm")
plt.ylabel("Metric Performance")
plt.title("Algorithm vs. Performance Against Iris Dataset")
plt.xticks(r + width/2, ["AdaBoost", "Random Forest"])
plt.legend()

plt.show()

wine_time = [np.mean(wine_ada_time), np.mean(wine_rf_time)]
wine_balanced_accuracy = [np.mean(wine_ada_balanced_accuracy), np.mean(wine_rf_balanced_accuracy)]
wine_accuracy = [np.mean(wine_ada_accuracy), np.mean(wine_rf_accuracy)]
wine_recall = [np.mean(wine_ada_recall), np.mean(wine_rf_recall)]
wine_precision = [np.mean(wine_ada_precision), np.mean(wine_rf_precision)]
wine_f1 = [np.mean(wine_ada_f1), np.mean(wine_rf_f1)]
wine_jaccard = [np.mean(wine_ada_jaccard), np.mean(wine_rf_jaccard)]
wine_time_err = [np.std(wine_ada_time), np.std(wine_rf_time)]
wine_balanced_accuracy_err = [np.std(wine_ada_balanced_accuracy), np.std(wine_rf_balanced_accuracy)]
wine_accuracy_err = [np.std(wine_ada_accuracy), np.std(wine_rf_accuracy)]
wine_recall_err = [np.std(wine_ada_recall), np.std(wine_rf_recall)]
wine_precision_err = [np.std(wine_ada_precision), np.std(wine_rf_precision)]
wine_f1_err = [np.std(wine_ada_f1), np.std(wine_rf_f1)]
wine_jaccard_err = [np.std(wine_ada_jaccard), np.std(wine_rf_jaccard)]

n = 2
r = np.arange(n)
width = 0.1

plt.bar(r, wine_time, width = width, edgecolor = "black", label = "Time")
plt.bar(r + width, wine_balanced_accuracy, width = width, edgecolor = "black", label = "Balanced Accuracy")
plt.bar(r + 2 * width, wine_accuracy, width = width, edgecolor = "black", label = "Accuracy")
plt.bar(r + 3 * width, wine_recall, width = width, edgecolor = "black", label = "Recall")
plt.bar(r + 4 * width, wine_precision, width = width, edgecolor = "black", label = "Precision")
plt.bar(r + 5 * width, wine_f1, width = width, edgecolor = "black", label = "F1-Score")

plt.xlabel("Algorithm")
plt.ylabel("Metric Performance")
plt.title("Algorithm vs. Performance Against Wine Dataset")
plt.xticks(r + width/2, ["AdaBoost", "Random Forest"])
plt.legend()

plt.show()

breast_cancer_time = [np.mean(breast_cancer_ada_time), np.mean(breast_cancer_rf_time)]
breast_cancer_balanced_accuracy = [np.mean(breast_cancer_ada_balanced_accuracy), np.mean(breast_cancer_rf_balanced_accuracy)]
breast_cancer_accuracy = [np.mean(breast_cancer_ada_accuracy), np.mean(breast_cancer_rf_accuracy)]
breast_cancer_recall = [np.mean(breast_cancer_ada_recall), np.mean(breast_cancer_rf_recall)]
breast_cancer_precision = [np.mean(breast_cancer_ada_precision), np.mean(breast_cancer_rf_precision)]
breast_cancer_f1 = [np.mean(breast_cancer_ada_f1), np.mean(breast_cancer_rf_f1)]
breast_cancer_jaccard = [np.mean(breast_cancer_ada_jaccard), np.mean(breast_cancer_rf_jaccard)]
breast_cancer_time_err = [np.std(breast_cancer_ada_time), np.std(breast_cancer_rf_time)]
breast_cancer_balanced_accuracy_err = [np.std(breast_cancer_ada_balanced_accuracy), np.std(breast_cancer_rf_balanced_accuracy)]
breast_cancer_accuracy_err = [np.std(breast_cancer_ada_accuracy), np.std(breast_cancer_rf_accuracy)]
breast_cancer_recall_err = [np.std(breast_cancer_ada_recall), np.std(breast_cancer_rf_recall)]
breast_cancer_precision_err = [np.std(breast_cancer_ada_precision), np.std(breast_cancer_rf_precision)]
breast_cancer_f1_err = [np.std(breast_cancer_ada_f1), np.std(breast_cancer_rf_f1)]
breast_cancer_jaccard_err = [np.std(breast_cancer_ada_jaccard), np.std(breast_cancer_rf_jaccard)]

n = 2
r = np.arange(n)
width = 0.1

plt.bar(r, breast_cancer_time, width = width, edgecolor = "black", label = "Time")
plt.bar(r + width, breast_cancer_balanced_accuracy, width = width, edgecolor = "black", label = "Balanced Accuracy")
plt.bar(r + 2 * width, breast_cancer_accuracy, width = width, edgecolor = "black", label = "Accuracy")
plt.bar(r + 3 * width, breast_cancer_recall, width = width, edgecolor = "black", label = "Recall")
plt.bar(r + 4 * width, breast_cancer_precision, width = width, edgecolor = "black", label = "Precision")
plt.bar(r + 5 * width, breast_cancer_f1, width = width, edgecolor = "black", label = "F1-Score")

plt.xlabel("Algorithm")
plt.ylabel("Metric Performance")
plt.title("Algorithm vs. Performance Against Breast Cancer Dataset")
plt.xticks(r + width/2, ["AdaBoost", "Random Forest"])
plt.legend()

plt.show()
