from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# LSTM is a variant of RNN
# LSTM here is merely a layer, and the type of model
# is a simple sequential model. Dense is a layer of
# regularly connected neurons

max_features = 20000

# fix the length of sequences as the NN accepts a fixed length as input
maxlen = 80 # cut texts after this number of words
# ( among top max_features most common words)

# Specify the num of batches during training
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Build NN
print('Build model...')
# Set a sequential type based model
model = Sequential()
# First layer of stacking so from 20000 goes to 128
# Can use own word embedding like Word2Vec or GloVe vectors too
model.add(Embedding(max_features, 128))
# Second layer LSTM, 128 is the total num of dimensions
# that the NN dealing with, recurrent_dropout prevents overfitting, 0.2
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# Third layer is a standard Dense layer with one input, which is the
# output from the LSTM
# activation function - how the NN decides input and what kind
# of output provides
model.add(Dense(1, activation='sigmoid'))

# try using diff optimizers and diff optomizer configs
# Run the compile method and use binary_crossentropy as loss and adam for optimizer
# All NN need a loss function and optimizer to learn
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=15, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)