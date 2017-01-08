# A seq2seq model for generating natural inference, using Word2vec
# Inspired by 'Generating Nautral Language Inference Chains', https://arxiv.org/abs/1606.01404
# Two pre-requisites:
# Retreive your own copy of the SNLI corpus files. I import and prepare using Pandas below.
# Train a Word2vec model and save (e.g. Gensim from the SNLI corpus or GoogleNews if you have the memory). I use Gensim below.
# As an alternative to Word2vec I've left commented out code for building your own vocab for TF's default embeddings. (is slow)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model

import gensim
import pandas as pd

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

tf.app.flags.DEFINE_integer("batch_size", 40, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.") # vs. 1024
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.") # vs. 3
tf.app.flags.DEFINE_integer("set1_vocab_size", 28181, "Premise vocab size.") # use whole vocab by default
tf.app.flags.DEFINE_integer("set2_vocab_size", 28181, "Entailment vocab size.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_string("train_dir", "/home/ubuntu/snli_training", "Training directory.") # vs /tmp
# tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.") # vs /tmp

tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]



print("fetch and prepare entailment data")

df = pd.read_csv('snli_1.0_dev.txt', sep="\t")
df = df[['gold_label', 'sentence1', 'sentence2']]
df = df[df.gold_label == "entailment"]

df2 = pd.read_csv('snli_1.0_train.txt', sep="\t")
df2 = df2[['gold_label', 'sentence1', 'sentence2']]
df2 = df2[df2.gold_label == "entailment"]

df3 = pd.read_csv('snli_1.0_test.txt', sep="\t")
df3 = df3[['gold_label', 'sentence1', 'sentence2']]
df3 = df3[df3.gold_label == "entailment"]

data_all = pd.concat([dfa, df2a, df3a], axis=0)

entail_source = filter(lambda x: type(x) is str, data_all.sentence1.values.tolist()) # a few odd entries
entail_target = filter(lambda x: type(x) is str, data_all.sentence2.values.tolist())
entail_combined = entail_source + entail_target
entail_zipped = zip(entail_source, entail_target)

print("load word2vec model")
w2v_model = gensim.models.Word2Vec.load('model')
w2v_X = w2v_model.syn0

# Alternatively, if you don't want to use Word2vec you can build your own vocab from corpus
# print("setting up vocab") # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
# vocab = {}
# vocab_list = []
# max_vocabulary_size = 40000
# for sent in entail_combined:
#     tokens = basic_tokenizer(sent)
#     for word in tokens: # optional normalize digits step
#         if word in vocab:
#             vocab[word] += 1
#         else:
#             vocab[word] = 1
#     vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
#     if len(vocab_list) > max_vocabulary_size:
#         vocab_list = vocab_list[:max_vocabulary_size]

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


print("importing SNLI vocab")
import csv
with open('snli_vocab.csv', 'rb') as f:
    reader = csv.reader(f)
    l = list(reader)
saved_vocab = [x[1:-1] for x in l[0]]
vocab_h = {}
for i,x in enumerate(saved_vocab):
    vocab_h[x] = i

rev_vocab_h = {v: k for k, v in vocab_h.iteritems()} # for decoding later on

print("convert sents to int id arrays")
entail_zipped_ids = []
for source, target in entail_zipped:
    words1 = basic_tokenizer(source)
    ids1 = [vocab_h.get(w, UNK_ID) for w in words1]

    words2 = basic_tokenizer(target)
    ids2 = [vocab_h.get(w, UNK_ID) for w in words2]

    entail_zipped_ids.append([ids1, ids2])
print("len entail zipped ids", len(entail_zipped))
print('done')







def read_data(zipped_ids_set):
  """
  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.

      why difference between source and target??

      _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
  """
  data_set = [[] for _ in _buckets]
  for source, target in zipped_ids_set:
      source_ids = source
      target_ids = target
      target_ids.append(data_utils.EOS_ID)
      for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids])
              break
  return data_set



def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

  model = seq2seq_model.Seq2SeqModel(
      FLAGS.set1_vocab_size,
      FLAGS.set2_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
      #dtype=dtype)

  # This should handle padding internally
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    #session.run(tf.initialize_all_variables())
    print("Also doing custom embedding tasks")
    embeddings = tf.Variable(tf.random_uniform(w2v_X.shape, minval=-0.1, maxval=0.1), trainable=False)
    session.run(tf.initialize_all_variables())
    session.run(embeddings.assign(w2v_X))
  return model


def train():
  #config = tf.ConfigProto()
  #config.gpu_options.allow_growth = True
  #config.gpu_options.per_process_gpu_memory_fraction = 0.4
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    model = create_model(sess, False)

    print("done creating model")

    train_raw = entail_zipped_ids[:160000]
    dev_raw = entail_zipped_ids[160000:]  # 30k test
    train_set = read_data(train_raw)
    dev_set = read_data(dev_raw)


    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()

      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint

      loss += step_loss / FLAGS.steps_per_checkpoint

      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)

          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")

          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()



# update this to use 
def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.

      words = basic_tokenizer(sentence)
      token_ids = [vocab_h.get(w, UNK_ID) for w in words]

      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]


      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_vocab_h[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
