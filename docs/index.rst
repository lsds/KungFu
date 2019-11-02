KungFu documentation
====================

KungFu aims to enable users to achieve easy, adaptive and fast distributed machine learning.
The following are the APIs we released by far.

Distributed optimizers
----------------------

KungFu provides optimizers that implement various distributed training algorithms.
These optimizers are used for transparently scaling out the training of
`tf.train.Optimizer <https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/Optimizer>`_
and `tf.keras.optimizers.Optimizer <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer>`_

.. automodule:: kungfu.tensorflow.optimizers
   :members:

Global variable initializers
----------------------------

KungFu provide various initializers to help you synchronize
the global variables of distributed training workers at the beginning of training.
These initializers are used with ``tf.session``, ``tf.estimator``, ``tf.gradient_tape``
and ``tf.keras``, respectively.

.. automodule:: kungfu.tensorflow.initializer
   :members:

Cluster management
------------------

When scaling out training, you often want to adjust the parameters of your
training program, for example, sharding the training dataset or scaling the learning rate
of the optimizer. This can be achieved using the following cluster management APIs.

.. automodule:: kungfu
   :members:

TensorFlow operators
--------------------

KungFu provides TensorFlow operators to help you realise
new distributed training optimizers.

.. automodule:: kungfu.tensorflow.ops
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
