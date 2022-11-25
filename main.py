import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_file_path = 'path to train file'
test_file_path  = 'path to test file'

df_train = pd.read_table(train_file_path, header = 0, names=['indicates', 'text'], usecols=['indicates', 'text'])
df_test = pd.read_table(test_file_path, header = 0, names=['indicates', 'text'], usecols=['indicates', 'text'])

df_train['indicates'] = df_train['indicates'].replace(['ham', 'spam'], [0, 1])
df_test['indicates'] = df_train['indicates'].replace(['ham', 'spam'], [0, 1])

train_data = tf.data.Dataset.from_tensor_slices((df_train['text'].values, df_train['indicates'].values))
test_data = tf.data.Dataset.from_tensor_slices((df_test['text'].values, df_test['indicates'].values))

tokenizer = tfds.deprecated.text.Tokenizer()

#vocabulary list from all data
vocabulary = set()

for text_tensor, label in test_data:
      tokens = tokenizer.tokenize(text_tensor.numpy())
      vocabulary.update(tokens)
vocab_size = len(vocabulary)

encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary)

def encode(text, label):
      encoded = encoder.encode(text.numpy())
      return encoded, label

def encode_map(text, label):
      encoded, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
      encoded.set_shape([None])

      return encoded, label

#encoded train and test data
train_encoded = train_data.map(encode_map)
test_encoded = test_data.map(encode_map)

# model training
BUFFER_SIZE = 10000
BATCH_SIZE = 32
train_dataset = (train_encoded.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [])))
test_dataset = (test_encoded.padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

model = tf.keras.Sequential([
      tf.keras.layers.Embedding(encoder.vocab_size, 32), 
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
history = model.fit(train_dataset, epochs = 10, validation_data = test_dataset, validation_steps = 30)

#loss and accuracy
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# function to predict messages based on model
# should return list containing prediction and label
def predict_message(pred_text):
      pred_text = encoder.encode(pred_text)
      pred_text = tf.cast(pred_text, tf.float32)
      prediction = model.predict(tf.expand_dims(pred_text, 0)).tolist()
      if prediction[0][0] < 0.5: 
            prediction = 'ham'
      else:
            prediction = 'spam'
      return prediction

def plot(history, metric):
      plt.plot(history.history[metric])
      plt.plot(history.history['val_' + metric], '')
      plt.xlabel('Epochs')
      plt.ylabel(metric)
      plt.legend([metric, 'val_' + metric])
      plt.show()

def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
