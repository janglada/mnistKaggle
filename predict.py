from keras.models import Sequential
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.utils import np_utils

test = pd.read_csv("./data/test.csv", delimiter = ",", dtype=np.float32).as_matrix()

# dimensions of our images.
img_width, img_height = 28, 28

X = test[:, 0:784]

X_test = X.reshape(X.shape[0], img_width, img_height, 1)




model = load_model('model_2.h5')


print("Generating test predictions...")
preds = model.predict_classes(X_test, batch_size=64, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp.csv")

# #
# #
# df3 = pd.DataFrame({'ImageId':np.arange(preds.flatten().shape[0])+1,'Label':preds.flatten()})
# df3.to_csv('submission.csv', index=False, header=True)