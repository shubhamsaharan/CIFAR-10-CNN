import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

np.random.seed(1000)


class Cifar10(object):
    def __init__(self, data_folder, n_training_files=5, image_width=32, image_height=32, n_components=3,
                 data_block_size=10000, training_set_size=50000, test_set_size=10000, n_classes=10):

        self.data_folder = data_folder
        self.n_training_files = n_training_files
        self.image_width = image_width
        self.image_height = image_height
        self.n_components = n_components
        self.data_block_size = data_block_size
        self.training_set_size = training_set_size
        self.test_set_size = test_set_size
        self.n_classes = n_classes

        self.training_set = np.ndarray(shape=(self.training_set_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)
        self.training_set_labels = np.ndarray(shape=(self.training_set_size, self.n_classes)).astype(np.float32)

        self.test_set = np.ndarray(shape=(self.test_set_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)
        self.test_set_labels = np.ndarray(shape=(self.test_set_size, self.n_classes)).astype(np.float32)

        print('Loading training data')

        for i in range(n_training_files):
            with open(self.data_folder + 'data_batch_' + str(i + 1), 'rb') as training_file:
                training_dict = cPickle.load(training_file, encoding='latin1')

                self.training_set[(self.data_block_size * i):(self.data_block_size * (i + 1)), :, :, :] = training_dict['data']. \
                    reshape((self.data_block_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)

                for idx, label in enumerate(training_dict['labels']):
                    self.training_set_labels[(self.data_block_size * i) + idx, :] = self.to_class(label)

        print('Loading test data')

        with open(self.data_folder + 'test_batch', 'rb') as test_file:
            test_dict = cPickle.load(test_file, encoding='latin1')

            self.test_set[0:self.data_block_size, :, :, :] = test_dict['data']. \
                reshape((self.data_block_size, self.n_components, self.image_width, self.image_height)).astype(np.float32)

            for idx, label in enumerate(test_dict['labels']):
                self.test_set_labels[idx, :] = self.to_class(label)

        with open(data_folder + 'batches.meta', 'rb') as label_file:
            self.label_dict = cPickle.load(label_file)
            self.label_names = self.label_dict['label_names']

        self.X_train, self.Y_train = (self.training_set / 255), self.training_set_labels
        self.X_test, self.Y_test = (self.test_set / 255), self.test_set_labels

    def to_class(self, label_idx):
        class_data = np.zeros(shape=self.n_classes).astype(np.float32)
        class_data[label_idx] = 1.0
        return class_data

    def to_label(self, class_vector):
        return self.label_names[np.argmax(class_vector)]

    def to_RGB(self, data):
        img = np.ndarray(shape=(self.image_width, self.image_height, self.n_components)).astype(np.uint8)

        for i in range(self.n_components):
            img[:, :, i] = data[i, :, :]

        return img

    def show_image(self, i, data_set='training'):
        if data_set == 'test':
            a_data_set = self.test_set
            a_data_set_labels = self.test_set_labels
        else:
            a_data_set = self.training_set
            a_data_set_labels = self.training_set_labels

        plt.imshow(self.to_RGB(a_data_set[i]))
        plt.show()
        return a_data_set_labels[i]


if __name__ == '__main__':
    print('CIFAR-10 Classification')

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    print('Fitting model')
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=128,
              shuffle=True,
              epochs=250,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

    print('Evalutating model')
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    print('Score: %1.3f' % scores[0])
    print('Accuracy: %1.3f' % scores[1])