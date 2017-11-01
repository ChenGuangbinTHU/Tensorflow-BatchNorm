import tensorflow as tf;    
import numpy as np;    
import matplotlib.pyplot as plt;    
  
v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(0))  
 
v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))  

v3 = tf.get_variable(name='v3', shape=[1], initializer=tf.constant_initializer(3))  

v4 = tf.get_variable(name='v4', shape=[1], initializer=tf.constant_initializer(4))  

  
with tf.Session() as sess:  
    sess.run(tf.initialize_all_variables())  
    tf.add_to_collection('loss', v1) 
    tf.add_to_collection('loss', v2)   
    print tf.get_collection('loss')  
    print sess.run(tf.get_collection('loss'))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())  
    tf.add_to_collection('loss', v3) 
    tf.add_to_collection('loss', v4)   
    print tf.get_collection('loss')  
    print sess.run(tf.get_collection('loss')) 