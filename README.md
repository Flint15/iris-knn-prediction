# Getting started

## Prerequisites

* scikit_learn

## Instalation

`pip install scikit-learn`

## Usage

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
`import KNeighborsClassifier` Import KNN algorithm

`load_iris` Import iris data set

`train_test_split` Needed to split our data

`accuracy_score` Evaluate accuracy

```python
iris = load_iris()
x, y = iris.data, iris.target
```
Load dataset

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

Method `train_test_split` split our data at a ratio of 1 to 5 for test and training data, respectively

x = 80% - x_train and 20% - x_test, y = 80% - y_train and 20% - y_test

`random_state=42` make shuffle of data reproducibility

```python
knn = KNeighborsClassifier(n_neighbors=3)
```
Initialize KNN classifier with 3 neighbors

```python
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
```

Train our model and make a prediction

```python
accuracy = accuracy_score(y_test, y_pred)
```

Evaluate accuracy
