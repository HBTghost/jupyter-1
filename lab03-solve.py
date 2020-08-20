import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

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
    res = self.img_2d[:, ::-1]
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

  def blur(self, box_size = 3, show=True):
    ml = lambda arr : np.insert(arr, arr.shape[1], 1, axis=1)[:, 1:]
    mr = lambda arr : np.insert(arr, 0, 1, axis=1)[:, :arr.shape[1]]
    mu = lambda arr : np.insert(arr, arr.shape[0], 1, axis=0)[1:, :]
    md = lambda arr : np.insert(arr, 0, 1, axis=0)[:arr.shape[0], :]

    la, ra, da, ua = ml(a), mr(a), md(a), mu(a)
    dla, ula, dra, ura = md(la), mu(la), md(ra), mu(ra)
    tmp = np.array([a, la, ra, da, ua, dla, ula, dra, ura])
    res = np.mean(tmp, axis=0, dtype=np.uint16)

    lr = [ml, mr]
    ud = [mu, md]

    a = self.img_2d
    pad = int(box_size / 2)
    
    alr, aud = [[a], [a]], [[], []]
    for i in range(pad):
      for j in range(2):
        alr[j].append(lr[j](alr[j][-1]))

    alr[0] = alr[0][1:]
    for i in range(2):
      for lr in alr[i]:
        for j in range(2):
          tmp = lr
          for k in range(pad):
            tmp = ud[j](tmp)
            aud[j].append(tmp)

    [x, y], [z, t] = alr, aud
    res = np.mean(np.concatenate([x, y, z, t]), axis=0, dtype=np.uint16)

    if show:
      self.add_to_show(res, "Blur")
    return res





if __name__ == '__main__':
  # lena = image("bird.jpg")
  # lena.bright(50)
  # lena.change_contrast(-200)
  # lena.grayscale()
  # lena.flip()
  # lena.blur2(5)
  # lena.show()
  # Open an already existing image

  imageObject = Image.open("bird.jpg")
  blurred = imageObject.filter(ImageFilter.BLUR)
  blurred.show()
