# kungfu

The ultimate distributed training framework for TensorFlow

## Build

Make sure you have tensorflow python installed.

```bash
# build a .whl package for release
pip3 wheel -vvv --no-index .

# install
pip3 install --no-index -U .
```

## Run

```bash
# FIXME: For Mac users, the following is required
# export DYLD_LIBRARY_PATH=$(python3 -c "import os; import kungfu; print(os.path.dirname(kungfu.__file__))")

python3 examples/kungfu-train.py
```
