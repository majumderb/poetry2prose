from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
#from keras.layers.core import RepeatVector, TimeDistributedDense, Activation
from keras.layers import TimeDistributed, Dense, Activation, Bidirectional, LSTM, RepeatVector
from seq2seq.cells import LSTMDecoderCell, AttentionDecoderCell
from sklearn.model_selection import train_test_split
from functools import reduce
import time
import numpy as np
import re

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def vectorize_stories(input_list, tar_list, word_idx, input_maxlen, tar_maxlen, vocab_size):
    x_set = []
    Y = np.zeros((len(tar_list), tar_maxlen, vocab_size), dtype=np.bool)
    for _sent in input_list:
        x = [word_idx[w] for w in _sent]
        x_set.append(x)
    for s_index, tar_tmp in enumerate(tar_list):
        for t_index, token in enumerate(tar_tmp):
            Y[s_index, t_index, word_idx[token]] = 1

    return pad_sequences(x_set, maxlen=input_maxlen), Y


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


######################################################################################################################################################
from collections import defaultdict
from copy import deepcopy
import string
import re

## Punctuation remover
translator = str.maketrans('', '', string.punctuation)
## regex for finding sloka number
regex = r"(\d+)\-(\d+)\-(\d+)"

# ANy is the prose, Slo is the sloka
slpAny = [item.split() for item in open('slpAny.txt').read().splitlines()]
slpSlo = [item.split() for item in open('slpSlo.txt').read().splitlines()]

sloAnv = defaultdict(dict)
for i,item in enumerate(slpSlo):
    if item[0].isnumeric() == False:
        print (item)

    sloAnv[int(item[0])]['slo'] = list()
    for stuff in item[1:]:
        match = re.search(regex, stuff)
        if match is not None:
            if match.start() != 0:
                stuff2 = re.sub(regex,'',stuff)

            elif match.end()!= len(stuff):
                stuff2 = re.sub(regex,'',stuff)
        else:
            stuff2 = stuff

        stuff2 = stuff2.strip().translate(translator).strip()
        if len(stuff2) > 0:
            sloAnv[int(item[0])]['slo'].append(stuff2)
for item in slpAny:
    if item[0].isnumeric() == False:
        print ('anv',item)

    sloAnv[int(item[0])]['anv'] = [stuff.strip().translate(translator).strip() for stuff in item[1:]]

print("Poetry is",sloAnv[0]['slo'])
print("Prose is",sloAnv[0]['anv'])

input_list = []
tar_list = []

for index in range(len(sloAnv)):
    #print(index)
    try:
        tar_list.append(sloAnv[index]['anv'])
        input_list.append(sloAnv[index]['slo'])
    except Exception:
        print("data missing at %r" %index)

# input_text = ['1 2 3 4 5'
#               , '6 7 8 9 10'
#               , '11 12 13 14 15'
#               , '16 17 18 19 20'
#               , '21 22 23 24 25']
# tar_text = ['one two three four five'
#             , 'six seven eight nine ten'
#             , 'eleven twelve thirteen fourteen fifteen'
#             , 'sixteen seventeen eighteen nineteen twenty'
#             , 'twenty_one twenty_two twenty_three twenty_four twenty_five']
#
# input_list = []
# tar_list = []
#
# for tmp_input in input_text:
#     input_list.append(tmp_input)
# for tmp_tar in tar_text:
#     tar_list.append(tmp_tar)

######################################################################################################################################################


vocab = sorted(reduce(lambda x, y: x | y, (set(tmp_list) for tmp_list in input_list + tar_list)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
input_maxlen = max(map(len, (x for x in input_list)))
tar_maxlen = max(map(len, (x for x in tar_list)))
output_dim = vocab_size
hidden_dim = 20

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Input max length:', input_maxlen, 'words')
print('Target max length:', tar_maxlen, 'words')
print('Dimension of hidden vectors:', hidden_dim)
print('Number of training stories:', len(input_list))
print('Number of test stories:', len(input_list))
print('-')
print('Vectorizing the word sequences...')
word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))
idx_to_word = dict((i + 1, c) for i, c in enumerate(vocab))
inputs_train, tars_train = vectorize_stories(input_list, tar_list, word_to_idx, input_maxlen, tar_maxlen, vocab_size)

np.random.seed(1337)  #for reproducibility
x_train, x_test, y_train, y_test = train_test_split(input_list, tars_train, test_size=0.2)

decoder_mode = 1
if decoder_mode == 3:
    encoder_top_layer = LSTM(hidden_dim, return_sequences=True)
else:
    encoder_top_layer = LSTM(hidden_dim)

if decoder_mode == 0:
    decoder_top_layer = LSTM(hidden_dim, return_sequences=True)
    decoder_top_layer.get_weights()
elif decoder_mode == 1:
    decoder_top_layer = LSTMDecoderCell(hidden_dim=hidden_dim, output_dim=hidden_dim
                                    , output_length=tar_maxlen, state_input=False, return_sequences=True)
# elif decoder_mode == 2:
#     decoder_top_layer = LSTMDecoder2(hidden_dim=hidden_dim, output_dim=hidden_dim
#                                     , output_length=tar_maxlen, state_input=False, return_sequences=True)
elif decoder_mode == 3:
    decoder_top_layer = AttentionDecoderCell(hidden_dim=hidden_dim, output_dim=hidden_dim
                                         , output_length=tar_maxlen, state_input=False, return_sequences=True)

en_de_model = Sequential()
en_de_model.add(Embedding(input_dim=vocab_size,
                          output_dim=hidden_dim,
                          input_length=input_maxlen))
en_de_model.add(encoder_top_layer)
if decoder_mode == 0:
    en_de_model.add(RepeatVector(tar_maxlen))
en_de_model.add(decoder_top_layer)

en_de_model.add(TimeDistributed(Dense(output_dim)))
en_de_model.add(Activation('softmax'))
print('Compiling...')
time_start = time.time()
en_de_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
time_end = time.time()
print('Compiled, cost time:%fsecond!' % (time_end - time_start))

model_file = "models/encoder_decoder_20170725"
best_model_cb = ModelCheckpoint(model_file, monitor="val_loss", verbose=1, save_best_only=True)
log_dir = "logs"
early_stopping_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')


for iter_num in range(5000):
    print('-' * 50)
    print('Iteration %r' %iter_num)
    print('-' * 50)
    en_de_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, nb_epoch=1, show_accuracy=True,
                     callbacks=[best_model_cb, early_stopping_cb])
    # out_predicts = en_de_model.predict(inputs_train)
    # for i_idx, out_predict in enumerate(out_predicts):
    #     predict_sequence = []
    #     for predict_vector in out_predict:
    #         next_index = np.argmax(predict_vector)
    #         next_token = idx_to_word[next_index]
    #         predict_sequence.append(next_token)
    #     print('Target output:', tar_text[i_idx])
    #     print('Predict output:', predict_sequence)
    #
    # print('Current iter_num is:%d' % iter_num)
