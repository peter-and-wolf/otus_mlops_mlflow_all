import random

import mlflow.entities
import numpy.typing as npt

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from data_source import extract_data


def get_metrics(y_test: npt.ArrayLike, y_pred: npt.ArrayLike):
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, zero_division=0)
  recall = recall_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)
  return accuracy, precision, recall, f1


def evaluate_model(X: npt.ArrayLike, 
                   y: npt.ArrayLike, 
                   model_name: str, 
                   alias: str) -> tuple[mlflow.entities.Run, float]:

  # load model from registry with name and tag
  model = mlflow.pyfunc.load_model(f'models:/{model_name}@{alias}')

  run_id = model.metadata.run_id
  run = mlflow.get_run(str(model.metadata.run_id))

  # print saved metrics
  print('Saved metrics:')
  print(f'  {run.data.metrics}')
  
  # calculate new metrcs
  y_pred = model.predict(X)
  a, p, r, f1 = get_metrics(y, y_pred)

  print('New metrics:')
  print(f"  {{'accuracy': {a}, 'precision': {p}, 'recall': {r}, 'f1': {f1}}}")

  return run, float(a)


def train_new_model(model_name: str,
                    X_train: npt.ArrayLike, 
                    y_train: npt.ArrayLike,
                    X_test: npt.ArrayLike,
                    y_test: npt.ArrayLike,
                    run: mlflow.entities.Run,
                    old_accuracy: float):

  model_uri = f'runs:/{run.info.run_id}/model'  
  model = mlflow.sklearn.load_model(model_uri)
  
  print('\nModel architecture:')
  print(f'  {model}')

  model.fit(X_train, y_train)

  # calculate new metrcs
  y_pred = model.predict(X_test)
  a, p, r, f1 = get_metrics(y_test, y_pred)

  print('Retrained metrics:')
  print(f"  {{'accuracy': {a}, 'precision': {p}, 'recall': {r}, 'f1': {f1}}}")

  with mlflow.start_run() as new_run:
    mlflow.log_params(run.data.params)
    mlflow.log_metrics({
      'accuracy': float(a),
      'precision': float(p),
      'recall': float(r),
      'f1': float(f1),
    })
    
    mlflow.sklearn.log_model(model, 'model')
    
    if a > old_accuracy:
      print("\nThe new model is better, let's put it in the registry...")
      mlflow.register_model(
        f'runs:/{new_run.info.run_id}/model', 
        f'{model_name}'
      )
  

if __name__ == '__main__':

  X, y = extract_data(n_informative=3, random_state=532, flip_y=.01, class_sep=random.uniform(.7, 1))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

  run, old_accuracy = evaluate_model(X_test, y_test, 'otus', 'staging')

  train_new_model('otus', X_train, y_train, X_test, y_test, run, old_accuracy)


 
  


  
  

  

