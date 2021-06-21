
# KungFu Failure Detection

In the distributed training process, each process sends "BEGIN" and "END" signals to the disaster recovery judgment machine respectively at the beginning and end of each step during the training process. When a process fails to send an "END" signal for a long time, it represents that it is "stuck" in the training process. Then the disaster recovery judgment machine returns the stuck signal. Kungfu will restart the training processes with the last available checkpoint.

## Project Architecture

<div align="center">
    <img src="Failure_detection_architecture.png" width="50%" height="30%"/>
</div>


## Implementation Architecture


<div align="center">
    <img src="Failure_detection_implementation.png" width="50%" height="30%"/>
</div>

KungFu currently implement failure detection and recovery in TF2_keras, Eager and TF1_session.

### KungFu send signal function
KungFu implement four signal send fuction: ``monitor_batch_begin()``, ``monitor_batch_end()``, ``monitor_epoch_end()`` and ``monitor_train_end()``.
1. ``monitor_batch_begin()`` is called at the beginning of each batch training. This function sends a "begin" signal to failure detection process.
2. ``monitor_batch_end()`` is used after the training of each batch. This function sends an "end" signal to failure detection process.
3. ``monitor_epoch_end()`` is used after the training of each epoch. This function tells failure detection process the training process has finished one epoch. After detecting the failures, the detection process can get how many epochs has finished and restart the training process to do the rest.
4. ``monitor_train_end()`` is used after training process. This function tells the detection process that the training is finished. Then the monitor process would stop.


### TF2_keras API

We use keras callback hook to achieve the signal sending. Here is an example.

```python
import tensorflow as tf
from kungfu.cmd import monitor_batch_begin,monitor_batch_end,monitor_train_end,monitor_epoch_end

class MonitorDetectionCallback(tf.keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs={}):
        monitor_batch_begin()
    def on_batch_end(self, batch, logs={}):
        monitor_batch_end()
    def on_epoch_end(self, epoch, logs={}):
        monitor_epoch_end()
 
...

def train_model(model, dataset, n_epochs=1, batch_size=5000, monitor = False, restart = 0):
    ...
    model.fit(x,
	      y,
	      batch_size=batch_size,
	      epochs=n_epochs,
	      validation_data=(dataset['x_val'], dataset['y_val']),
	      verbose=2,
	      callbacks=[BroadcastGlobalVariablesCallback(),checkpoint,MonitorDetectionCallback()])
        
...

if __name__ == '__main__':
    main()
    monitor_train_end()
```

You can find a full example in [TensorFlow Keras](https://github.com/DingtongHan/Kungfumonitormlbd/blob/main/examples/Failure_recovery_examples/tf2_mnist_keras.py).

### Eager signal send

In Eager and TF1_session, we both train the model batch by batch. We can call ``monitor_batch_begin()`` and ``monitor_batch_end()`` at the beginning and end of each for-loop. Then we can calculate when an epoch finish and call ``monitor_epoch_end()``. Here is an example of eager training, you can find the full example in [Eager](https://github.com/DingtongHan/Kungfumonitormlbd/blob/main/examples/Failure_recovery_examples/eager.py).

```python
import tensorflow as tf
from kungfu.cmd import monitor_batch_begin,monitor_batch_end,monitor_train_end,monitor_epoch_end

def train_model(model, dataset, n_epochs=1, batch_size=5000, monitor = False, restart = 0):
    ...
    for local_step, (images, labels) in enumerate(ds):
        if args.monitor:
            monitor_batch_begin()
        global_step.assign_add(1)
        trained_samples.assign_add(current_cluster_size() * args.batch_size)
        loss_value = training_step(model, loss, opt, images, labels)
        step = int(global_step)
        print('step: %d loss: %f' % (step, loss_value))
        if args.monitor:
            if trained_samples >= MNIST_DATA_SIZE * (epochs+1):
                model.save(savepath)
                monitor_epoch_end()
                epochs = epochs + 1
            monitor_batch_end()
        if trained_samples >= total_samples:
            break
        
...

if __name__ == '__main__':
    check_tf_version()
    print('main started')
    main()
    print('main finished')
    monitor_train_end()
```

### TF1_session signal send

The implementation method is similar to eager training, you can call ``monitor_batch_begin()`` and ``monitor_batch_end()`` at the beginning and end of each for range. You can find the full example in [TF1_session](https://github.com/DingtongHan/Kungfumonitormlbd/blob/main/examples/Failure_recovery_examples/tf1_mnist_session.py).

```python
import tensorflow as tf
from kungfu.cmd import monitor_batch_begin,monitor_batch_end,monitor_train_end,monitor_epoch_end

def train_mnist(sess, x, y_, train_op, test_op, optimizer, dataset, n_epochs=1, batch_size=5000, monitor=False, restart=0):
    ...
    for step in range(n_steps):
        if monitor:
            monitor_batch_begin()
        xs = dataset['training_set']['x'][offset:offset + batch_size]
        y_s = dataset['training_set']['y'][offset:offset + batch_size]
        offset = (offset + batch_size * n_shards) % training_set_size
        sess.run(train_op, {
            x: xs,
            y_: y_s,
        })
        # log the validation accuracy
        if step % log_period == 0:
            training_acc_dataset = dict()
            training_acc_dataset['x'] = xs
            training_acc_dataset['y'] = y_s
            result = test_mnist(sess, x, y_, test_op, training_acc_dataset)
            print('training accuracy: %f' % result)
            result = test_mnist(sess, x, y_, test_op,
                                dataset['validation_set'])
            print('validation accuracy: %f' % result)
        if monitor:
            if step%step_per_epoch == 0 and step !=0:
                saver.save(sess,savepath)
                print(step/step_per_epoch)
                monitor_epoch_end()
            monitor_batch_end()
        
...

def main():
    args = parse_args()
    optimizer = build_optimizer(name=args.kf_optimizer,
                                batch_size=args.batch_size)
    x, y_, train_op, test_op = build_model(optimizer)
    mnist = load_mnist(args.data_dir)
    
    with tf.Session() as sess:
        train_mnist(sess, x, y_, train_op, test_op, optimizer, mnist,
                    args.n_epochs, args.batch_size, args.monitor,args.restart)
        result = test_mnist(sess, x, y_, test_op, mnist['test_set'])
        print('test accuracy: %f' % result)
        # save_all(sess, 'final')
    if args.monitor:
        monitor_train_end()
```
### Failure detection

KungFu has designed a complete detection example [Here](https://github.com/DingtongHan/Kungfumonitormlbd/blob/main/srcs/go/cmd/kungfu-recovery/monitor.go). The principle is that when the detection receive "BEGIN" signal, detection process will record a time stamp. If we still do not receive "END" signal after 10 seconds (for Mnist training) from receiving "BEGIN" signal. Failure detection process will assume the training process is stucked. Then it will report "some machine died" and finish process. ``times[]`` is a list record the time of receiving "begin" signal. ``trainend[]`` is a list which store whether a training process has finished. ``epochs[]`` stores the epoch number each training process has finished.

```golang

//solve different signal
func (h *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    var msg Message
    err := json.NewDecoder(r.Body).Decode(&msg)
    if err != nil {
        return
    }
    datas := strings.Split(string(msg.Key), ":")
    intva, err := strconv.Atoi(datas[1])
    if err != nil{
    }
    if datas[0] == "trainend" {
        trainend[intva] = 1
    }
    if datas[0] == "begin" {
        times[intva] = time.Now().Unix()
    }
    if datas[0] == "end" {
        times[intva] = 0
    }
    if datas[0] == "epoch" {
        epochs[intva] = epochs[intva]+1
    }
}

...

        //judge whether receive 'end' signal after 10 seconds
        for i:= 0; i < *machines; i++ {
            if trainend[i] == 1{
                trainendflag = trainendflag + 1
            }
            if a := time.Now().Unix() - times[i]; a > 10{
                if times[i] != 0{
                    min := findmin(epochs)
                    flag := "some machine died:" + strconv.Itoa(min)
                    fmt.Println(flag)
                    os.Exit(0)
                }
            }
        }
...

```

## File change

### Add file

examples/Failure_recovery_examples/eager.py (an example of failure detection in eager training)

examples/Failure_recovery_examples/tf1_mnist_session.py (an example of failure detection in tf1_session training)

examples/Failure_recovery_examples/tf2_mnist_keras.py (an example of failure detection in tf2_keras training)

srcs/go/cmd/kungfu-recovery/monitor.go (an example of detect failures)

docs/Failure_detection_architecture.png (the overall architecture)

docs/Failure_detection_implementation.png (the implementation architecture we designed)

docs/Failures_definition.png

### Modified file

srcs/cpp/include/kungfu.h (add the definition of send function)

srcs/cpp/srcs/kungfu.cpp (add the definition of send function)

srcs/go/kungfu/runner/flags.go (add monitor property)

srcs/go/cmd/kungfu-run/app/kungfu-run.go (add monitor property)

srcs/go/kungfu/runner/simple.go (add restart code)

srcs/go/libkungfu-comm/cmds.go (add the definition of send signals)

srcs/go/utils/iostream/iostream.go (add the "some machine died" signal detection)

srcs/go/utils/iostream/stdio.go (add the "some machine died" signal detection and return signal to main run process)

srcs/go/utils/runner/local.go (add run monitor process and return machine died signal)

srcs/python/kungfu/cmd/__init__.py (add the definition of signal send function)
