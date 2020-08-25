import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm

class wine:
  def __init__(self, filename="wine.csv"):
    self.df = pd.read_csv(filename, sep=';')
    self.B = self.df['quality'].to_numpy()

    preA = self.df.drop(['quality'], axis=1)
    self.title_A = preA.columns
    self.A = preA.to_numpy()

  def lirem(self):
    return np.linalg.pinv(self.A) @ self.B

  def cross(self):
    bestA = np.array([
      cross_val_score(
        svm.SVC(kernel='linear', C=1),
        self.A[:, [i]],
        self.B,
        cv=5
      ).mean()
      for i in range(self.A.shape[1])
    ]).argmax(axis=0)

    return self.title_A[bestA], (np.linalg.pinv(self.A[:, [bestA]]) @ self.B)[0]

  def clf(self):
    is_good = lambda x : 1 if x > 5 else 0
    newB = np.vectorize(is_good)(self.B)

    bestA = np.array([
      cross_val_score(
        svm.SVC(kernel='linear', C=1),
        self.A[:, [i]],
        newB,
        cv=5
      ).mean()
      for i in range(self.A.shape[1])
    ]).argmax(axis=0)
    
    return self.title_A[bestA], (np.linalg.pinv(self.A[:, [bestA]]) @ newB)[0]


if __name__ == '__main__':
  w = wine()
  for x in w.lirem():
    print ("{:f}".format(x))
  print (w.cross())
  print (w.clf())
