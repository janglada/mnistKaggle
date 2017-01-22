import pandas as pd
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import skimage.transform as transform
from scipy import ndimage
from skimage import img_as_float, img_as_ubyte




df = pd.read_csv("./data/train.csv", delimiter = ",", dtype=np.float32)

df =  df.sample(n=10000)






# dimensions of our images.
img_width, img_height = 28, 28


temp = []
for index, row in df.iterrows():
   rowData  = row.as_matrix();
   X_row    = rowData[1:785]
   Y_row    = rowData[0]
   X_row    = X_row.reshape(img_width, img_height)

   x_dilation = ndimage.grey_dilation(X_row, size=(2,2))
   x_erosion  = ndimage.grey_erosion(X_row,  size=(2,2))

   affine = transform.AffineTransform(shear=np.deg2rad(np.random.uniform(-20, 20)),
                                      rotation=np.deg2rad(np.random.uniform(-20, 20)))

   X_affine = img_as_ubyte(transform.warp(img_as_float(X_row / 255.0), affine))

   # plt.figure(1)
   # plt.subplot(411)
   # plt.imshow(X_row, cmap='Greys', interpolation='nearest')
   #
   # plt.subplot(412)
   # plt.imshow(x_dilation, cmap='Greys', interpolation='nearest')
   #
   # plt.subplot(413)
   # plt.imshow(X_affine, cmap='Greys', interpolation='nearest')
   #
   # plt.subplot(414)
   # plt.imshow(x_erosion, cmap='Greys', interpolation='nearest')
   # plt.show()

   x_dilation = x_dilation.reshape(784)
   X_affine = X_affine.reshape(784)
   x_erosion = x_erosion.reshape(784)

   x_dilation = np.insert(x_dilation, 0,Y_row)
   X_affine = np.insert(X_affine, 0, Y_row)
   x_erosion = np.insert(x_erosion, 0, Y_row)


   # print X_affine
   # print x_erosion
   # df_transformation.loc[df_transformation.shape[0]] = x_dilation
   # df_transformation.loc[df_transformation.shape[0]] = X_affine
   # df_transformation.loc[df_transformation.shape[0]] = x_erosion

   temp.append(x_dilation)
   temp.append(X_affine)
   temp.append(x_erosion)


   # print index

df_transformation   = pd.DataFrame(temp, columns=df.columns)

df_transformation.to_csv("train_transformation.csv", index=False, header=True)



# # print df_transformation
# #
#
#
#
# X = train[:, 1:785]
# X = X.reshape(X.shape[0], img_width, img_height)
#
#
#
# arr = X[0]
# print X.shape
# print arr.shape
#
#
# #
# plt.figure(1)
# plt.subplot(411)
# plt.imshow(arr, cmap='Greys',  interpolation='nearest')
#
#
#
# plt.subplot(412)
# plt.imshow(ndimage.grey_erosion(arr, size=(2,2)).astype(arr.dtype), cmap='Greys',  interpolation='nearest')
#
# plt.subplot(413)
# plt.imshow(ndimage.grey_dilation(arr, size=(2,2)).astype(arr.dtype), cmap='Greys',  interpolation='nearest')
# # plt.show()
#
#
# affine = transform.AffineTransform(shear=np.deg2rad(np.random.uniform(-20,20)), rotation=np.deg2rad(np.random.uniform(-15,15)))
#
# arr = transform.warp(img_as_float(arr/255.0), affine)
#
# plt.subplot(414)
# plt.imshow(arr, cmap='Greys',  interpolation='nearest')
# plt.show()
#
# print arr.reshape(784)













