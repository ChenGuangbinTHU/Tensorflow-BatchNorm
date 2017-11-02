import tensorflow as tf;    
import numpy as np;    
import matplotlib.pyplot as plt;    
  
A = np.arange(96)

B = np.reshape(A,[3,2,4,4])
C = np.reshape(A,[24,4]) 
with tf.Session() as sess:
    data_tf = tf.convert_to_tensor(C, np.float32)
    print(sess.run(tf.nn.moments(data_tf,(0))))
    # print(np.mean(C,[1]))