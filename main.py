from sklearn.datasets import load_svmlight_file
import os

DATA_DIR='./data'

data_files = os.listdir(DATA_DIR)

f = data_files[0]
print('Filename ', f)


print('Getting data')

X, y = load_svmlight_file(os.path.join(DATA_DIR, f))

print('X shape ', X.shape)
print('Y shape ', y.shape)