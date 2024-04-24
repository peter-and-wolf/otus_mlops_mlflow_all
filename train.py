import warnings
warnings.filterwarnings('ignore')

import itertools

import mlflow
import mlflow.sklearn

import numpy as np
import numpy.typing as ntp

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from data_source import extract_data

if __name__ == '__main__':

  X, y = extract_data(n_informative=3, random_state=463, flip_y=.4)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

  grid = {
    'C': np.logspace(-3, 3, 5),
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'penalty': ['l1', 'l2'],
  }

  for C, solver, penalty in itertools.product(*(grid[p] for p in grid.keys())):
    
    try:
      model = make_pipeline(
        StandardScaler(), 
        LogisticRegression(
          C=C, 
          solver=solver, 
          penalty=penalty, 
          max_iter=100
        )
      )

      # Fit model
      model.fit(X_train, y_train)

      # Make predictions
      y_pred = model.predict(X_test)

      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred, zero_division=0)
      recall = recall_score(y_test, y_pred, zero_division=0)
      f1 = f1_score(y_test, y_pred, zero_division=0)

      print(f'LR(C={C}, solver={solver}, penalty={penalty}): accuracy={accuracy}')

      with mlflow.start_run():
        mlflow.log_params({
          'C': C,
          'solver': solver,
          'penalty': penalty,
        })
        mlflow.log_metrics({
          'accuracy': float(accuracy),
          'precision': float(precision),
          'recall': float(recall),
          'f1': float(f1),
        })

        mlflow.sklearn.log_model(model, 'model')

    except ValueError:
      pass
    


