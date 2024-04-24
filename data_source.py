import numpy.typing as npt

from sklearn.datasets import make_classification

N_SAMPLES = 10_000
N_FEATURES = 22

def extract_data(n_informative: int, random_state: int, flip_y: float = .9, class_sep: float = .7) -> tuple[npt.ArrayLike, npt.ArrayLike]:
  assert(n_informative <= N_FEATURES)
  X, y = make_classification(
    random_state=random_state,
    n_samples=N_SAMPLES, 
    n_features=N_FEATURES, 
    n_informative=n_informative, 
    n_classes=2, 
    flip_y=flip_y,       # The higher the value, the more the labels are shuffled
    class_sep=class_sep  # The lower the value, the closer the classes
  )
  return X, y