# https://allennlp.org/elmo

from allennlp.commands.elmo import ElmoEmbedder
from numpy import linalg as LA
import math
# load weights
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'


elmo = ElmoEmbedder(options_file, weight_file)
while True:
    x = input('input sentence one: ').split()
    y = input('input sentence two: ').split()
    vectors1 = elmo.embed_sentence(x)
    vectors2 = elmo.embed_sentence(y)

    word_level_distancs = 1 / (1 + math.exp(-(abs(LA.norm(vectors1[0]) - LA.norm(vectors2[0])))))
    syntax_level_distancs = 1 / (1 + math.exp(-(abs(LA.norm(vectors1[1]) - LA.norm(vectors2[1])))))
    semantics_level_distancs = 1 / (1 + math.exp(-(abs(LA.norm(vectors1[2]) - LA.norm(vectors2[2])))))

    print('Word level distance', word_level_distancs)
    print('Syntax level distance', syntax_level_distancs)
    print('Semantics level distance', semantics_level_distancs)
    # TODO: try different measures
