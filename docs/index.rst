KungFu documentation
====================

KungFu aims to make distributed machine learning easy, adaptive and scalable.

Getting started
===============

We try to keep it as simple as possible to install, deploy and run KungFu.
KungFu does not require extra deployments like parameter servers
or heavy dependencies like OpenMPI and NCCL as in Horovod.
KungFu can run on your laptop, your desktop
and your server, with and without GPUs.
Please follow the instruction in the README to install KungFu.

Examples
========

We provide various examples
to show how to use KungFu with various TensorFlow objects and Keras models.

Session
-------

TensorFlow Session is a low-level but powerful interface that
allows you to compile a static graph for iterative training.
Session is the core for TensorFlow 1 programs. To enable KungFu,
you need to wrap your ``tf.train.Optimizer`` in a KungFu
distributed optimizer, and
use ``BroadcastGlobalVariablesOp`` to broadcast global variables
at the first step of your training.

.. code-block:: python

    import tensorflow as tf

    # Build model...
    loss = ...
    opt = tf.train.AdamOptimizer(0.01)

    # KungFu Step 1: Wrap tf.optimizer in KungFu optimizers
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    opt = SynchronousSGDOptimizer(opt)

    # Make training operation
    train_op = opt.minimize(loss)

    # Train your model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # KungFu Step 2: ensure distributed workers start with consistent states
        from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
        sess.run(BroadcastGlobalVariablesOp())

        for step in range(10):
            sess.run(train_op)

You can find the full training example: `TensorFlow 1 Session <https://github.com/lsds/KungFu/blob/master/examples/tf1_mnist_session.py>`_

Estimator
---------

TensorFlow Estimator is the high-level API for TensorFlow 1 programs.
To enable KungFu, you need to wrap your ``tf.train.Optimizer`` in a KungFu
distributed optimizer, and
register ``BroadcastGlobalVariablesHook`` as a hook for the estimator.

.. code-block:: python

    import tensorflow as tf

    def model_func():
        loss = ...
        opt = tf.train.AdamOptimizer(0.01)

        # KungFu Step 1: Wrap tf.optimizer in KungFu optimizers
        from kungfu.tensorflow.optimizers import SynchronousAveragingOptimizer
        opt = SynchronousAveragingOptimizer(opt)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=opt.minimize(loss))

    # KungFu Step 2: register the broadcast global variables hook
    from kungfu.tensorflow.initializer import BroadcastGlobalVariablesHook
    hooks = [BroadcastGlobalVariablesHook()]

    estimator = tf.estimator.Estimator(model_fn=model_func,
                                       model_dir=FLAGS.model_dir)

    for _ in range(10):
        estimator.train(input_fn=train_data, hooks=hooks)

You can find the full training example: `TensorFlow 1 Estimator <https://github.com/lsds/KungFu/blob/master/examples/tf1_mnist_estimator.py>`_

GradientTape
------------

TensorFlow 2 supports eager execution for the ease of building dynamic models.
The core of the eager execution is the ``tf.GradientTape``.
To enable KungFu, you need to wrap your ``tf.train.Optimizer`` in a KungFu
distributed optimizer, and use ``broadcast_variables`` to broadcast global
variables at the end of the first step of training.


.. code-block:: python

    import tensorflow as tf

    # Build the dataset...
    dataset = ...

    # Build model...
    loss = ...
    opt = tf.keras.optimizers.SGD(0.01)

    # KungFu Step 1: Wrap tf.optimizer in KungFu optimizers
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    opt = SynchronousSGDOptimizer(opt)

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = mnist_model(images, training=True)
            loss_value = loss(labels, probs)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

        # KungFu Step 2: broadcast global variables
        if first_batch:
            from kungfu.tensorflow.initializer import broadcast_variables
            broadcast_variables(mnist_model.variables)
            broadcast_variables(opt.variables())

        return loss_value

    for batch, (images, labels) in enumerate(dataset.take(10000)):
        loss_value = training_step(images, labels, batch == 0)

You can find the full training example: `TensorFlow 2 GradientTape <https://github.com/lsds/KungFu/blob/master/examples/tf2_mnist_gradient_tape.py>`_

TensorFlow Keras
----------------

Keras has become the high-level training API for
TensorFlow since 1.11 and has become the default interface in TensorFlow 2.
To enable KungFu, you need to wrap your ``tf.train.Optimizer`` in a KungFu
distributed optimizer, and use ``BroadcastGlobalVariablesCallback``
as a callback for Keras model.

.. code-block:: python

    import tensorflow as tf

    # Build dataset...
    dataset = ....

    # Build model...
    model = tf.keras.Sequential(...)
    opt = tf.keras.optimizers.SGD(0.01)

    # KungFu Step 1: Wrap tf.optimizer in KungFu optimizers
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer
    opt = SynchronousSGDOptimizer(opt)

    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                  optimizer=opt,
                  metrics=['accuracy'])

    # KungFu Step 2: Register a broadcast callback
    from kungfu.tensorflow.initializer import BroadcastGlobalVariablesCallback
    model.fit(dataset,
              steps_per_epoch=500,
              epochs=1,
              callbacks=[BroadcastGlobalVariablesCallback()])

Here are two full training examples:
`TensorFlow 1 Keras <https://github.com/lsds/KungFu/blob/master/examples/tf1_mnist_keras.py>`_
and  `TensorFlow 2 Keras <https://github.com/lsds/KungFu/blob/master/examples/tf2_mnist_keras.py>`_


Keras
-----

KungFu can be used with Keras in the same way as the above TensorFlow Keras example.
You simply pass an extra `with_keras` flag to both KungFu optimizers and
Keras callback to tell KungFu you are using Keras not TensorFlow.
Here is a full Keras training example: `Keras <https://github.com/lsds/KungFu/blob/master/examples/keras_mnist.py>`_

KungFu APIs
===========

KungFu has the high-level optimizer APIs that
allows you to transparently scale out training.
It also has a low-level API that allows an easy implementation
of distributed training strategies.
The following is the public API we released so far.

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
the global variables of distributed training workers at
the beginning of training. These initializers are used
with ``tf.session``, ``tf.estimator``, ``tf.GradientTape``
and ``tf.keras``, respectively.

.. automodule:: kungfu.tensorflow.initializer
   :members:

Cluster management
------------------

When scaling out training, you often want to adjust
the parameters of your training program, for example,
sharding the training dataset or scaling the learning rate
of the optimizer. This can be achieved using the following
cluster management APIs.

.. automodule:: kungfu.python
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
