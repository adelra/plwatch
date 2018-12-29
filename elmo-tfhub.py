# use either ELMO or BERT
# https://allennlp.org/elmo

# imports
import tensorflow as tf
import tensorflow_hub as hub

# load weights

with tf.Graph().as_default():
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(["the cat is on the mat", "dogs are in the fog"], signature="default", as_dict=True)["elmo"]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))
print('model loaded')
# get the vectors for the input strings

# compare the distance, for now the cosine similarity

#TODO: try different measures

# output the chance of being similar
