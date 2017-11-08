import tensorflow as tf
from nets import mobilenet_v1




input = tf.placeholder("float32", shape=[1, 1000, 800, 3])

net, features =    mobilenet_v1.mobilenet_v1_base(input)

shape = net.get_shape().as_list()
print("Network shape", shape)

print("There are %d endpoints" % len(features))

for key in sorted(features):
    shape = features[key].get_shape().as_list()
    print("\t%s shape:" % key, shape)