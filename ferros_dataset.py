from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

def dipole(x, y, z, dx, dy, dz, mx, my, mz):
    R = (x - dx)**2 + (y - dy)**2 + (z - dz)**2
    return (3.0*(x - dx) * ((x - dx)*mx + (y - dy)*my + (z - dz)*mz) / R**2.5 - mx/R**1.5,
            3.0*(y - dy) * ((x - dx)*mx + (y - dy)*my + (z - dz)*mz) / R**2.5 - my/R**1.5,
            3.0*(z - dz) * ((x - dx)*mx + (y - dy)*my + (z - dz)*mz) / R**2.5 - mz/R**1.5)

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0) #normalize!
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._image_size = images.shape[1] #correct?

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def image_size(self):
    return self._image_size

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 900
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def make_data_set(image_size = 50,
                  set_size = 5000,
                  test_size = 500,
                  validation_size=100,
                  seed=None):
  all_images = []
  all_labels = [0]*(set_size+test_size) #заглушка
  print('genereate ferros data set')
  for i in range(set_size+test_size):
    img = []
    dz = -5.0*numpy.random.ranf()-1.1
    dx = numpy.random.randint(0, image_size)
    dy = numpy.random.randint(0, image_size)
    mz = numpy.random.randint(0, 10000)
    mx = numpy.random.randint(-10000, 10000)
    my = numpy.random.randint(-10000, 10000)
    for i in range(image_size):
      for j in range(image_size):
        img.append(dipole(i, j, 0, dx, dy, dz, mx, my, mz)[2] + numpy.random.normal(0, 1))
    all_images.append(img)
  all_images=numpy.array(all_images)
  all_labels=numpy.array(all_labels)
  print ('ferros data set done')
  #if not 0 <= validation_size <= len(train_images):
  #  raise ValueError(
  #      'Validation size should be between 0 and {}. Received: {}.'
  #      .format(len(train_images), validation_size))
  
  test_images = all_images[:test_size]
  test_labels = all_labels[:test_size]

  train_images = all_images[test_size:]
  train_labels = all_labels[test_size:]

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]


  options = dict(dtype=dtypes.float32, reshape=False, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)
