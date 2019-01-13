import tensorflow as tf

print('Version: %s' % tf.__version__)
print('Lib: %s' % tf.sysconfig.get_lib())
print('Include: %s' % tf.sysconfig.get_include())
print('Compile Flags: %s' % tf.sysconfig.get_compile_flags())
print('Link Flags: %s' % tf.sysconfig.get_link_flags())
