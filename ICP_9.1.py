import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
import numpy
data = pd.read_csv('data/Sentiment.csv')
data = data[['text','sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print(score)
print(acc)
print(model.metrics_names)
model.save('model.h5')
mod = load_model('model.h5')

print(mod.summary())
txt = [['A lot of good things are happening. We are respected again throughout the world, and thats a great '
         'thing.@realDonaldTrump']]
max_data = pd.DataFrame(txt, index=range(0, 1, 1), columns=list('t'))
max_data['t'] = max_df['t'].apply(lambda x: x.lower())
max_data['t'] = max_df['t'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
features = 2000
tokenizer = Tokenizer(num_words=features, split=' ')
tokenizer.fit_on_texts(max_data['t'].values)
X = tokenizer.texts_to_sequences(max_data['t'].values)
X = pad_sequences(X, maxlen=28)

out = mod.predict(X)
print(out)
print(numpy.where(max(out[0])), ":", (max(out[0])))
print(numpy.argmax(out))
print(mod.summary())