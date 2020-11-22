from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode(sequence):
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

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Build Network
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Validate Approach
x_validate = x_train[:1000]
y_validate = one_hot_train_labels[:1000]
partial_x_train = x_train[1000:]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_validate, y_validate))

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

graph_data(train_data=loss, 
           validate_data=val_loss,
           train_label='Training loss',
           validate_label='Validation loss',
           title='Training and validation loss',
           filename='Newswires_TrainingAndValidationLoss')

graph_data(train_data=accuracy, 
           validate_data=val_accuracy,
           train_label='Training accuracy',
           validate_label='Validation accuracy',
           title='Training and validation accuracy',
           filename='Newswires_TrainingAndValidationAccuracy')

# Build Improved Network
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_validate, y_validate))

results = model.evaluate(x_test, one_hot_test_labels)
print(results)