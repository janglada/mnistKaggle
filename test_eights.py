import pandas as pd
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import skimage.transform as transform
from scipy import ndimage
from skimage import img_as_float
df = pd.read_csv("./data/train.csv", delimiter = ",", dtype=np.float32, nrows=50)

eights =  df[df['label'] == 9]
train = eights.as_matrix()


# dimensions of our images.
img_width, img_height = 28, 28


X = train[:, 1:785]
X = X.reshape(X.shape[0], img_width, img_height)



arr = X[0]
print X.shape
print arr.shape


#
plt.figure(1)
plt.subplot(411)
plt.imshow(arr, cmap='Greys',  interpolation='nearest')



plt.subplot(412)
plt.imshow(ndimage.grey_erosion(arr, size=(2,2)).astype(arr.dtype), cmap='Greys',  interpolation='nearest')

plt.subplot(413)
plt.imshow(ndimage.grey_dilation(arr, size=(2,2)).astype(arr.dtype), cmap='Greys',  interpolation='nearest')
# plt.show()


affine = transform.AffineTransform(shear=np.deg2rad(np.random.uniform(-20,20)), rotation=np.deg2rad(np.random.uniform(-15,15)))

arr = transform.warp(img_as_float(arr/255.0), affine)

plt.subplot(414)
plt.imshow(arr, cmap='Greys',  interpolation='nearest')
plt.show()











