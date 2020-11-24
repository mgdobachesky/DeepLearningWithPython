from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sequence(sequence):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence])

def graph_data(train_data, validate_data, train_label, validate_label, title, filename):
    epochs = range(1, len(train_data) + 1)
    plt.clf()

    plt.plot(epochs, train_data, 'bo', label=train_label)
    plt.plot(epochs, validate_data, 'b', label=validate_label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(filename)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Prepare Data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Build Network
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Validate Approach
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]
x_validation = x_train[:10000]
y_validation = y_train[:10000]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_validation, y_validation))

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

# Plot Validation Scores
graph_data(train_data=loss, 
           validate_data=val_loss,
           train_label='Training loss',
           validate_label='Validation loss',
           title='Training and validation loss',
           filename='MovieReviews_TrainingAndValidationLoss')

graph_data(train_data=accuracy, 
           validate_data=val_accuracy,
           train_label='Training accuracy',
           validate_label='Validation accuracy',
           title='Training and validation accuracy',
           filename='MovieReviews_TrainingAndValidationAccuracy')

# Build Improved Network
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print(results)