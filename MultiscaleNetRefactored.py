
# coding: utf-8

# In[17]:

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# This notebook is a refactored version of a Multiscale Convolutional Network solution for NeuroHack at Yandex

# In[4]:

import random

from tqdm import tqdm
import pandas as pd
import h5py
import numpy as np
from scipy.signal import resample


# In[5]:

train_path = "./hackaton_data/train.h5"
test_path = "./hackaton_data/test.h5"
model_dump_path = "./hackaton_data/convnet-multiscale-true-01988"

slice_len = 1125


# Read train data into a dict of (subject_id, (X, y)) pairs

# In[6]:

subjects = {}
with h5py.File(train_path, "r") as data_file:
    for subject, subject_data in data_file.items():
        X = subject_data["data"][:]
        y = subject_data["labels"][:][0]
        subjects[subject] = (X, y)


# We need to split our data into train and local validation sets. For local validation we select random slices of slice_len. They may overlap with slices in train set, this split is not optimal.

# In[7]:

from sklearn.model_selection import train_test_split

def train_val_split(X, y):
    start_indices = list(range(0, len(X) - slice_len))
    y = y[:len(start_indices)]
    indices_train, indices_test, _, _ = train_test_split(start_indices, y)
    return {"train_ind": indices_train, "val_ind": indices_test, "X": X, "y": y}


# In[8]:

for subject in subjects:
    X, y = subjects[subject][0], subjects[subject][1]
    X = X.T
    subjects[subject] = train_val_split(X, y)


# Define utility function to convert class labels to one-hot encoding

# In[9]:

def to_onehot(y):
    onehot = np.zeros(3)
    onehot[y] = 1
    return onehot


# Define a funtion that will select random subject and find a random subsequence of consistent class

# In[10]:

def generate_slice(slice_len, val=False):
    subject_data = random.choice(list(subjects.values()))
    if val is True:
        indices, y, X = subject_data["val_ind"], subject_data["y"], subject_data["X"]
    else:
        indices, y, X = subject_data["train_ind"], subject_data["y"], subject_data["X"]
    
    while True:
        slice_start = random.choice(indices)
        slice_end = slice_start + slice_len
        slice_x = X[slice_start:slice_end]
        slice_y = y[slice_start:slice_end]
        
        if len(set(slice_y)) == 1:
            return slice_x, to_onehot(slice_y)


# Define a generator that will yield batches of resampled input time series and according class labels in infinite loop

# In[11]:

def data_generator(batch_size, slice_len, val=False):
    while True:
        batch_x = []
        batch_y = []
        
        for i in range(0, batch_size):
            x, y = generate_slice(slice_len, val=val)
            batch_x.append(x)
            batch_y.append(y)
            
        y = np.array(batch_y)
        
        x_256 = np.array([resample(i, 256) for i in batch_x])
        x_500 = np.array([resample(i, 500) for i in batch_x])
        x = np.array([i for i in batch_x])
        yield ([x_256, x_500, x], y)


# Lets build a neural network. Import all needed layers and keras utilities

# In[12]:

from keras.layers import Convolution1D, Dense, Dropout, Input, merge, GlobalMaxPooling1D
from keras.models import Model, load_model
from keras.optimizers import RMSprop


# This function builds a base neural net model that performs feature extraction

# In[13]:

def get_base_model(input_len, fsize):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_seq = Input(shape=(input_len, 24))
    nb_filters = 150
    convolved = Convolution1D(nb_filters, fsize, border_mode="same", activation="tanh")(input_seq)
    processed = GlobalMaxPooling1D()(convolved)
    compressed = Dense(150, activation="tanh")(processed)
    compressed = Dropout(0.3)(compressed)
    compressed = Dense(150, activation="tanh")(compressed)
    model = Model(input=input_seq, output=compressed)            
    return model


# Build and compile a graph with 3 inputs and one output

# In[14]:

input256_seq = Input(shape=(256, 24))
input500_seq = Input(shape=(500, 24))
input1125_seq = Input(shape=(1125, 24))
    
base_network256 = get_base_model(256, 4)
base_network500 = get_base_model(500, 7)
base_network1125 = get_base_model(1125, 10)

embedding_256 = base_network256(input256_seq)
embedding_500 = base_network500(input500_seq)
embedding_1125 = base_network1125(input1125_seq)
    
merged = merge([embedding_256, embedding_500, embedding_1125], mode="concat")
out = Dense(3, activation='softmax')(merged)
    
model = Model(input=[input256_seq, input500_seq, input1125_seq], output=out)
    
opt = RMSprop(lr=0.005, clipvalue=10**6)
model.compile(loss="categorical_crossentropy", optimizer=opt)


# This cell will train the model from scratch, lets load it from the model dump instead (take a look at next code cell)!

# In[15]:

from keras.callbacks import EarlyStopping

nb_epoch = 100000
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
samples_per_epoch = 4000

model.fit_generator(data_generator(batch_size=50, slice_len=slice_len), samples_per_epoch, nb_epoch, 
                    callbacks=[earlyStopping], verbose=1, nb_val_samples=2000,
                    validation_data=data_generator(batch_size=50, slice_len=slice_len, val=True))


# In[ ]:

model = load_model("hackaton_data/convnet-multiscale-true-01988")


# In[ ]:

model.summary()


# Read test data into a nested structure with multiple chunks for each subject

# In[ ]:

with h5py.File("hackaton_data/test.h5", "r") as data_file:
    test = {}
    for subject, subject_data in data_file.items():
        test[subject] = {}
        for chunk_id, chunk in data_file[subject].items():
            test[subject][chunk_id] = chunk[:]


# In[ ]:

test['subject_0']['chunk_0'].shape


# In[ ]:

# utility function that performs resampling of input timeseries 
def multiscale(chunk):
    resampled_256 = resample(chunk, 256)
    resampled_500 = resample(chunk, 500)
    return [resampled_256, resampled_500, chunk]


# Make prediction for test data

# In[ ]:

df = []
for subj in test:
    for chunk in tqdm(test[subj]):
        data = {}
        data["subject_id"] = int(subj.split("_")[-1])
        data["chunk_id"] = int(chunk.split("_")[-1])
        arr = test[subj][chunk].T
        preds = model.predict([np.array([i]) for i in multiscale(arr)])[0]
        data["class_0_score"] = preds[0]
        data["class_1_score"] = preds[1]
        data["class_2_score"] = preds[2]
        for i in range(0, 1125):
            data["tick"] = i
            df.append(data.copy())
df = pd.DataFrame(df)
df = df[["subject_id", "chunk_id", "tick", "class_0_score",
         "class_1_score","class_2_score"]]


# In[ ]:

df.head()


# In[ ]:

# save submission to .csv
df.to_csv("submission.csv")

