import tensorflow as tf
a =tf.constant([1.0,2.0],name = 'a')
b =tf.constant([1.0,2.0],name = 'b')
result =tf.add(a,b,name = 'add')
# print(result)
# print(a)
# print(b)
result = a+b
print(result)
#use session to get the final ture result
print(tf.Session.run(result))
print(tf.Session().run(result))
print(a.graph is tf.get_default_graph())
print(b.graph is tf.get_default_graph())#tf.get_default_graph函数可以回去当先默认的计算图
#--------------------------------------------------------------------------------------
import tensorflow as tf
g1 = tf.Graph() #tf.Graph() produce a new graph
with g1.as_default():
    v = tf.get_variable(
        'v', initializer  = tf.zeros_initializer()(shape = [1]))
g2  = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
        'v',initializer  = tf.ones_initializer()(shape = [1]))
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse = True):
        print(sess.run(tf.get_variable('v')))
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse = True):
        print(sess.run(tf.get_variable("v")))
g = tf.Graph()
with g.device('/gpu:0'):
    result = a+b