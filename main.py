import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split
import tensorflow as tf

#comment block: ctrl + /
apple = pd.read_csv("AAPL.csv")

#30 day moving average
apple['SMA30'] = apple['Adj Close Price'].rolling(window=30).mean() #mean of previous 30 entries at a time
apple['SMA100'] = apple['Adj Close Price'].rolling(window=100).mean() #mean of previous 100 entries at a time
#print(apple.size)
del apple['Close Price']

plt.figure(figsize=(12, 5))
plt.plot(apple['Adj Close Price'], label='Apple')
plt.plot(apple['SMA30'], label='30 day moving average')
plt.plot(apple['SMA100'], label='100 day moving average')
plt.title('Apple Adj Close Price History')
plt.xlabel("May 27,2014 - May 25,2020 ")
plt.ylabel("Adj Close Price USD ($)")
plt.legend(loc="upper left")
plt.show()
apple.reset_index(drop=True, inplace=True)

#first n rows :data.iloc[:n]);; every nth row data.iloc[::n])

#split data

date_change = '%m/%d/%Y'
#apple['Date'] = apple.index
apple['Date'] = pd.to_datetime(apple['Date'], format = date_change)
# model using  keras

##set y is target and x is feature
target = apple.pop('Adj Close Price')
apple.pop('Date')

###error concert df to tensor; error on dtype
# tf.convert_to_tensor(apple)
#tf.reshape(apple, [-1, 6])
dataset = tf.data.Dataset.from_tensor_slices((apple.values, target.values))
dataset = dataset.shuffle(
    13590, seed=None, reshuffle_each_iteration=None
)
#tf.reshape(dataset, [-1, 6])
 #all = tf.reshape(dataset, [-1, 8])
#test = dataset.take(1000)
#train = dataset.skip(1000)
# train_data = tf.data.Dataset.from_tensor_slices((train))
# test_data = tf.data.Dataset.from_tensor_slices((test))
print(dataset)
model = tf.keras.Sequential(

[
        tf.keras.layers.Dense(12, input_dim=6,activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ]
)
# # compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#fit the keras model on the dataset


# inpTensor = train((6,))
#
# #create the layers and pass them the input tensor to get the output tensor:
# hidden1Out = Dense(units=4)(inpTensor)
# hidden2Out = Dense(units=4)(hidden1Out)
# finalOut = Dense(units=1)(hidden2Out)
#
# #define the model's start and end points
# model = Model(inpTensor,finalOut)

model.fit(apple.values,target.values
          ,
          epochs=150, batch_size=10)
#evaluate the keras model
_, accuracy = model.evaluate(test)
print('Accuracy: %.2f' % (accuracy*100))

#model.fitDataset(dataset, {epochs: 5})