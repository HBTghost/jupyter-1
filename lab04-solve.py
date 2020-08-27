import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm

class wine:
  def __init__(self, filename="wine.csv"):
    self.df = pd.read_csv(filename, sep=';')
    self.df.fillna(0, inplace=True)
    self.df.drop_duplicates(keep=False, inplace=True)

    self.B = self.df['quality'].to_numpy()
    preA = self.df.drop(['quality'], axis=1)
    self.title_A = preA.columns
    self.A = preA.to_numpy()

  def cross_val_scores(self):
    return np.array([
      cross_val_score(
        svm.SVC(kernel='linear', C=1),
        self.A[:, [i]],
        self.B,
        cv=5
      ).mean()
      for i in range(self.A.shape[1])
    ])

  def lireg_all(self):
    x = np.linalg.pinv(self.A) @ self.B
    return [(self.title_A[i], x[i]) for i in range(len(x))]

  def lireg_best(self):
    bestA = self.cross_val_scores().argmax(axis=0)
    x = np.linalg.pinv(self.A[:, [bestA]]) @ self.B
    return self.title_A[bestA], x[0]

  def my_lireg(self):
    pre_newA = self.df.drop([
      'quality',
      'residual sugar',
      'chlorides',
      'free sulfur dioxide',
      'density'
    ], axis=1)
    x = np.linalg.pinv(pre_newA.to_numpy()) @ self.B
    return [(pre_newA.columns[i], x[i]) for i in range(len(x))]

if __name__ == '__main__':
  w = wine()

  print ('Cau a. Su dung toan bo 11 dac trung de bai cung cap')
  for x in w.lireg_all():
    print (x)

  print ('Cau b. Su dung duy nhat 1 dac trung cho ket qua tot nhat')
  print (w.lireg_best())

  print ('Cau c. Su dung mo hinh cua em')
  for x in w.my_lireg():
    print ("{}".format(x))
