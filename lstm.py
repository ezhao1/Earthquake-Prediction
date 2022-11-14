import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, SimpleRNN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Any results you write to the current directory are saved as output.
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
      #features are also included in the input data
    X = [] #input
    Y = [] #target
    i = 0
    while i < len(data) - seqlen - 1:
        numfeatures = 7
        temp = np.zeros((seqlen, numfeatures + 1))
        allNums = []
        sum = 0
        max = float("inf") *-1
        min = float("inf")

        for x in range(0, seqlen):
            temp[x][0] = (data[i+x][0])
            allNums.append(data[i+x][0])
            sum += data[i+x][0]
            if data[i+x][0] > max:
                max = data[i+x][0]
            if data[i+x][0] < min:
                min = data[i+x][0]
        mean = sum / seqlen
        pct25 = np.percentile(allNums, 25)
        pct50 = np.percentile(allNums, 50)
        pct75 = np.percentile(allNums, 75)
        for n in range(0, seqlen):
            temp[n][1] = mean
            temp[n][2] = min
            temp[n][3] = max
            temp[n][4] = sum
            temp[n][5] = pct25
            temp[n][6] = pct50
            temp[n][7] = pct75
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
print("aaaa")
combined = list(zip(X_total, Y_total))
np.random.shuffle(combined)
X_total[:], Y_total[:] = zip(*combined)
n = int(n/unroll)
X_train = X_total[:n]
print("Shape of xtrain", X_train.shape)
print(X_train[0])
Y_train = Y_total[:n]

X_test = X_total[n:]
Y_test = Y_total[n:]


Y_train2 = convertToCriticalLevel(Y_train)
Y_test2 = convertToCriticalLevel(Y_test)





print("Shape of training inputs", X_train.shape)
print("Shape of training targets", Y_train2.shape)

print("Shape of testing inputs", X_test.shape)
print("Shape of testing targets", Y_test2.shape)



def build_lstm_model(input_size, output_size, unrolling_steps):
    model = Sequential()
    model.add(LSTM(100, use_bias=True, dropout=0.2,
                   batch_input_shape=(None, unrolling_steps, input_size)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_size))
    model.add(Activation('selu'))
    model.compile(loss='cosine_proximity', optimizer='Nadam', metrics=['accuracy'])
    return model



lstm = build_lstm_model(8, 3, unroll)

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
