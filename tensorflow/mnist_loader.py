"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return x_training data, which has the shape=(50000,784) and 
       y_training_data which has shape (50000,10)
       for example x_training[0] is 784 dim vector, and y_training[0]
       is the corresponding desired output of the net 
       ( y_training[i]=1 means that y_training[0] corresponds to the number i.)

       x_validation, y_validation, x_test, y_test are analogous. wit 10000 entries.

    """
    tr_d, va_d, te_d = load_data()
    x_training=tr_d[0]
    x_validation=va_d[0]
    x_test=te_d[0]

    y_training=np.zeros(shape=(len(x_training),10))
    y_validation=np.zeros(shape=(len(x_validation),10))
    y_test=np.zeros(shape=(len(x_test),10))

    for i in range(0,len(y_training)):
        y_training[i][tr_d[1][i]]=1.0
    for i in range(0,len(y_validation)):
        y_validation[i][va_d[1][i]]=1.0
    for i in range(0,len(y_test)):
        y_test[i][te_d[1][i]]=1.0
    
    return x_training,y_training,x_validation,y_validation,x_test,y_test

