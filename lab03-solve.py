import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

def get_labels(img_1d, centroids, max_ram):
  buf = int(15000000 * max_ram / centroids.shape[0])
  buf = buf if buf > 0 else 1
  size = int(img_1d.shape[0] / buf)

  res = [np.array([np.linalg.norm(img_1d[buf * i : buf * (i + 1)] - centroid, axis=1) for centroid in centroids]).argmin(axis=0) for i in range(size)]
  remain = np.array([np.linalg.norm(img_1d[size*buf:]-centroid, axis=1) for centroid in centroids]).argmin(axis=0)
  if remain.shape[0] > 0:
    res.append(remain)

  return np.concatenate(res, axis=0)


def get_centroids(img_1d, labels, centroids):
  size = centroids.shape[0]
  res = []
  for i in range(size):
    mask = np.where(labels == i)
    centroid = np.random.randint(255, size=3)
    if len(mask[0]) > 0:
      pre_group_centroids = np.take(img_1d, mask, 0)
      centroid = np.mean(pre_group_centroids, dtype=np.int32, axis=1)
    res.append(centroid)
  return np.vstack(res)


def kmeans(img_1d, k_clusters, max_iter, init_centroids, max_ram):
  size = img_1d.shape[0]
  delta = k_clusters / (k_clusters + 50)
  pre_labels = np.zeros(size, dtype=np.int32)
  cur_labels = np.zeros(size, dtype=np.int32)

  centroids = np.array([])
  if init_centroids == 'random':
    centroids = np.random.randint(255, size=(k_clusters, 3))
  else:
    centroids = np.array([img_1d[np.random.randint(size)] for i in range(k_clusters)])
    
  for iterator in range(max_iter):
    cur_labels = get_labels(img_1d, centroids, max_ram)
    if abs(np.mean(cur_labels) - np.mean(pre_labels)) < delta:
      break
    pre_labels[:] = cur_labels
    centroids = get_centroids(img_1d, cur_labels, centroids)

  return cur_labels, centroids


def blur(img_1d, k_clusters, max_iter, init_centroids, max_ram):
  labels, centroids = kmeans(img_1d, k_clusters, max_iter, init_centroids, max_ram)
  return np.take(centroids, labels, axis=0)


def blur_image():
  # Handle input
  img_name = input("Enter image name (image and this python file must be in same folder): ")
  img_2d = np.array(Image.open(img_name))

  k_clusters = input("Enter number of clusters (hw: k = (3, 5, 7); default: k = 16): ")
  max_iter = input("Enter max iterators (default: max_iter = 300): ")
  init_centroids = input("Enter init centroids (default: init_centroids = 'random'): ")
  max_ram = input("Enter max RAM in GB want to use (default: max_ram = 1 - program will only use maximun 1GB): ")

  is_hw = False

  if k_clusters == "hw":
    is_hw = True
    k_clusters = [3, 5, 7]
  elif k_clusters in ["", "default"]:
    k_clusters = [16]
  else:
    k_clusters = [int(k_clusters)]

  if max_iter != '':
    max_iter = int(max_iter)
  else:
    max_iter = 300

  if init_centroids != 'random':
    init_centroids = 'in_pixels'

  if max_ram != '':
    max_ram = float(max_ram)
  else:
    max_ram = 1

  # Blur process
  w, h, d = tuple(img_2d.shape)
  img_1d = np.reshape(img_2d, (w * h, d))
  img_1d_blurs = [blur(img_1d, k, max_iter, init_centroids, max_ram) for k in k_clusters]
  img_2d_blurs = [np.reshape(img_1d_blur, (w, h, d)) for img_1d_blur in img_1d_blurs]

  # Show images
  titles = ['Original image ({} color)'.format(np.unique(img_1d, axis=0).shape[0])]
  titles.extend(['Blur image ({} colors)'.format(i) for i in k_clusters])
  k_clusters.insert(0, np.unique(img_1d, axis=0).shape[0])
  img_2d_blurs.insert(0, img_2d)

  for i in range(len(k_clusters)):
    plt.figure(i+1, figsize=(6, 4))
    plt.clf()
    plt.title(titles[i])
    plt.imshow(img_2d_blurs[i])

  plt.show()

class image:
  def __init__(self, name):
    self.name = name
    self.img_2d = np.array(Image.open(name))
    self.w, self.h, self.d = tuple(self.img_2d.shape)
    self.quantity = 0
    self.add_to_show(self.img_2d)
  
  def add_to_show(self, img_2d, name = "Original"):
    self.quantity += 1
    plt.figure(self.quantity)
    plt.clf()
    plt.title(name)
    plt.imshow(img_2d)
    
  def show(self):
    plt.show()

  def bright(self, brightness, show=True):
    trunc = lambda pixel : max(0, min(255, pixel + brightness))
    res = np.vectorize(trunc)(self.img_2d)
    if show:
      self.add_to_show(res, "Brightness {}{}".format("+" if brightness > 0 else "", brightness))
    return res

  def change_contrast(self, level, show=True):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    contrast = lambda pixel : max(0, min(255, int(128 + factor * (pixel - 128))))
    res = np.vectorize(contrast)(self.img_2d)
    if show:
      self.add_to_show(res, "Change contrast level {}".format(level))
    return res

  def grayscale(self, show=True):
    tmp = np.asarray(np.dot(self.img_2d, [.3, .59, .11]), dtype=np.uint8)
    res = np.dstack([tmp]*3)
    if show:
      self.add_to_show(res, "Grayscale")
    return res

  def flip(self, show=True):
    res = np.fliplr(self.img_2d)
    if show:
      self.add_to_show(res, "Flip")
    return res

  def merge(self, img, show=True):
    my_img_gray = self.grayscale(False)
    your_img_gray = img.grayscale(False)
    res = np.add(my_img_gray, your_img_gray, dtype=np.uint16)
    if show:
      self.add_to_show(res, "Merge with {}".format(img.name))
    return res

  def get_elem(self, i, j):
    i = max(0, min(i, self.w-1))
    j = max(0, min(j, self.h-1))
    return self.img_2d[i][j]

  def box(self, box_size, i, j):
    half = int(box_size / 2)
    tmp = [self.get_elem(i+x-half, j+y-half) for y in range(box_size) for x in range(box_size)]
    return np.mean(np.vstack(tmp), axis=0, dtype=np.uint16)
    
  def blur(self, box_size=3, show=True):
    res = np.zeros((self.w, self.h, self.d), dtype=np.uint8)
    for i in range(self.w):
      for j in range(self.h):
        res[i][j] = self.box(box_size, i, j)
    if show:
      self.add_to_show(res, "Blur")
    return res



if __name__ == '__main__':
  lena = image("lena.png")
  # lena.bright(50)
  # lena.change_contrast(-200)
  # lena.grayscale()
  # lena.flip()
  bear = image("pandemic.jpeg")
  lena.merge(bear)
  # lena.blur(9)
  lena.show()
  # Open an already existing image

  # imageObject = Image.open("lena.png")
  # blurred = imageObject.filter(ImageFilter.BLUR)
  # blurred.show()
