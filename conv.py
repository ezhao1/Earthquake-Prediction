import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, SimpleRNN
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
size = 10000000
train = pd.read_csv("/scratch/tschrad1/train.csv", nrows=size,
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

n = train.head(5)
print(n)
print(train.shape)


arr = train.values
print(arr[0][1])



def create_sequences(data, seqlen):
    #creating sequences of acoustic data inputs for the LSTM to read

    X = [] #input
    Y = [] #target
    i = 0
    while i < len(data) - seqlen - 1:
        temp = np.zeros(seqlen)
        for x in range(0, seqlen):
            temp[x] = data[i+x][0]


        X.append(temp)
        Y.append(data[i+seqlen][1])
        i += seqlen


    return np.array(X), np.array(Y)

n = int(0.9 * len(arr))#testing and training split
unroll = 100

def convertToCriticalLevel(array):
    #converts targets to three levels of earthquake imminence

    newArray = []
    for index, time in enumerate(array):
        newArray1 = np.zeros(3)
        if time < 3:
            newArray1[0] = 1.0
            newArray.append(newArray1)
        elif time >= 3 and time < 6:
            newArray1[1] = 1.0
            newArray.append(newArray1)
        else:
            newArray1[2] = 1.0
            newArray.append(newArray1)

    newArrayNP = np.array(newArray)

    return newArrayNP

X_total, Y_total = create_sequences(arr, unroll)
print(X_total.shape)
print(X_total[0])
#randomization
combined = list(zip(X_total, Y_total))
np.random.shuffle(combined)
X_total[:], Y_total[:] = zip(*combined)
n = int(n/unroll)
X_train = X_total[:n]

print(X_train.shape)
#shaping the inputs for the convnet
n_features = 1
n_seq = 2
n_steps = int(unroll/n_seq)
X_train = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))

Y_train = Y_total[:n]

X_test = X_total[n:]
Y_test = Y_total[n:]
X_test = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))

Y_train2 = convertToCriticalLevel(Y_train)
Y_test2 = convertToCriticalLevel(Y_test)





print("Shape of training inputs", X_train.shape)
print("Shape of training targets", Y_train2.shape)

print("Shape of testing inputs", X_test.shape)
print("Shape of testing targets", Y_test2.shape)



def build_lstm_model(input_size, output_size, unrolling_steps, n_steps, n_features):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=100, kernel_size=12, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100, use_bias=True, dropout=0.2,
                   batch_input_shape=(None, unrolling_steps, input_size)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_size))
    model.add(Activation('selu'))
    model.compile(loss='cosine_proximity', optimizer='Nadam', metrics=['accuracy'])
    return model



lstm = build_lstm_model(8, 3, unroll, n_steps, n_features)

# Train the model
history = lstm.fit(X_train, Y_train2, batch_size=32, epochs=2)
#print(train["acoustic_data"][0])
import matplotlib
import matplotlib.pyplot as plt
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["acc"], label="accuracy")
plt.show()


loss, accuracy = lstm.evaluate(X_test, Y_test2)
print("\nloss", loss, "accuracy", accuracy)
