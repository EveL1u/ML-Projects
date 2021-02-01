# -*- coding: utf-8 -*-

# Import modules
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# For windows laptops:
import matplotlib 
matplotlib.use('agg')
%matplotlib auto

import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

"""### Data preparation

#### Import data
"""

def load_data():
    # Import MNIST dataset from openml
    dataset = fetch_openml('mnist_784', version=1, data_home=None)

    # Data preparation
    raw_X = dataset['data']
    raw_Y = dataset['target']
    return raw_X, raw_Y

raw_X, raw_Y = load_data()


def clean_data(raw_X, raw_Y):

    cleaned_X = raw_X.astype('float32')
    cleaned_X /= 255
    
    num_classes = 10
    cleaned_Y = keras.utils.to_categorical(raw_Y, num_classes)
    
    return cleaned_X, cleaned_Y

cleaned_X, cleaned_Y = clean_data(raw_X, raw_Y)

"""#### Data split

- Split data into a train set (50%), validation set (20%) and a test set (30%).
"""

def split_data(cleaned_X, cleaned_Y):

    X_train, X_test, Y_train, Y_test = train_test_split(cleaned_X, cleaned_Y, test_size=0.3, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=2/7, random_state=42)
    
    return X_val, X_test, X_train, Y_val, Y_test, Y_train

X_val, X_test, X_train, Y_val, Y_test, Y_train = split_data(cleaned_X, cleaned_Y)

"""#### Plot data with matplotlib
"""

def viz_data(X_train):
    X_train_sample = X_train[:10,]
    Y_train_sample = Y_train[:10,]

    
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train_sample[i].reshape(28,28), cmap='Greys', interpolation='none')
        plt.title("Class {}".format(Y_train_sample[i].argmax()))
        plt.subplots_adjust(wspace=1)
        plt.show()
        plt.savefig('viz_data.png')

viz_data(X_train)

"""### Model

#### Neural network structure
- For this example network, we'll use 2 hidden layers
- Layer 1 has 128 nodes, a dropout rate of 20%, and relu as its activation function
- Layer 2 has 64 nodes, a dropout rate of 20%, and relu as its activation function
- The last layer maps back to the 10 possible MNIST class, and uses softmax as the activation


"""

def build_model():
    
      model = Sequential([
        Dense(128, activation='relu', name='layer1', input_shape=(784, )),
        Dropout(0.2),
        Dense(64, activation='relu', name='layer2'),
        Dropout(0.2),
        Dense(10, activation='softmax', name='layer3')
      ])
     
      return model

model = build_model()

"""# Model compilation
- Use categorical crossentropy as loss function

# Model training
- Use a batch size of 128, and train for 12 epochs
- Use verbose training, include validation data

"""

def compile_model(model):

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # use Adam optimizer as it's adaptive and generally outperforms SGD, along with a list of other benefits 

    return model

def train_model(model, X_train, Y_train, X_val, Y_val):

    history = model.fit(X_train, Y_train, batch_size = 128, epochs = 12, verbose=1, validation_data=(X_val, Y_val))
    return model, history


model = compile_model(model)
model, history = train_model(model, X_train, Y_train, X_val, Y_val)

"""# Model evaluation
- Show the performance on the test set
- Identify a few images the model classifies incorrectly
"""

def eval_model(model, X_test, Y_test):

    score = model.evaluate(X_test, Y_test, verbose=1)
    test_loss = score[0] 
    test_accuracy = score[1]
    print('Test Loss:', '%.4f' % test_loss)
    print('Test Accuracy:', '%.4f' % test_accuracy)

    pred = model.predict(X_test)
    i=0
    mis_class=[]

    for i in range(len(Y_test)):
        if(not Y_test[i].argmax()==pred[i].argmax()):
          mis_class.append(i)
        if(len(mis_class)==6):  # get 6 mis-classified examples
          break
    
    plt.figure()
    for i, incorrect in enumerate(mis_class):
      i += 1
      plt.subplot(2, 3, i)
      plt.imshow(X_test[incorrect].reshape(28,28), cmap='Greys', interpolation='none')
      plt.title("Predicted {}, Class {}".format(pred[incorrect].argmax(), Y_test[incorrect].argmax()))
      plt.subplots_adjust(wspace=1, hspace=0.6)
      plt.suptitle('Incorrectly Classified Sample:', size=15, weight=5)
      plt.show()
      plt.savefig('incorret_sample.png')

    return test_loss, test_accuracy

test_loss, test_accuracy = eval_model(model, X_test, Y_test)

model.summary()