import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud
from textblob import TextBlob

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import word2vec
from collections import Counter
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          TextClassificationPipeline)
import joblib
from tqdm import tqdm
import shap
import contractions

data = pd.read_csv("./twitter_parsed_dataset.csv").drop(columns=['id', 'index', "Annotation"])
data.columns = ['text', 'label']
data.head(25)

data['text'].sample(1).values[0]
data['label'].value_counts().plot(kind = 'bar', color = sns.color_palette('pastel'))
plt.xticks([0,1],['positive', 'negative'], rotation = 0);


only_english = set(nltk.corpus.words.words())
def clean_text(text):
    
    sample = text
    sample = " ".join([x.lower() for x in sample.split()])
    sample = re.sub(r"\S*https?:\S*", '', sample) #links and urls
    sample = re.sub('\[.*?\]', '', sample) #text between [square brackets]
    sample = re.sub('\(.*?\)', '', sample) #text between (parenthesis)
    sample = re.sub('#', ' ', sample) #remove hashtags
    sample = ' '.join([x for x in sample.split() if not x.startswith('@')]) # remove mentions with @
    sample = " ".join([contractions.fix(x) for x in sample.split()])  # fixes contractions like you're to you are
    sample = re.sub('[%s]' % re.escape(string.punctuation), ' ', sample) #punctuations
    sample = re.sub('\w*\d\w', '', sample)
    sample = re.sub(r'\n', ' ', sample) #new line character
    sample = re.sub(r'\\n', ' ', sample) #new line character
    sample = re.sub("[''""...“”‘’…]", '', sample) #list of quotation marks
    sample = " ".join(x.strip() for x in sample.split()) #strips whitespace
    sample = re.sub(r', /<[^>]+>/', '', sample)    #HTML attributes
    
    sample = ' '.join(list(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is not None, sample.split()))) #languages other than english
    
    sample = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE).sub(r'', sample) #emojis and symbols
    sample = sample.strip()
    sample = " ".join([x.strip() for x in sample.split()])
    return sample

data['cleaned_text'] = data['text'].apply(lambda x: clean_text(str(x)))
data


stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def get_wordnet_pos(word):
    
    treebank_tag = nltk.pos_tag([word])[0][1]
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def correct_text(text, stem=False, lemma=False, spell=False):
    if lemma and stem:
        raise Exception('Either stem or lemma can be true, not both!')
        return text
    
    sample = text
    
    #removing stopwords
    sample = sample.lower()
    sample = [word for word in sample.split() if not word in stops]
    sample = ' '.join(sample)
    
    if lemma:
        sample = sample.split()
        sample = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word.lower())) for word in sample]
        sample = ' '.join(sample)
        
    if stem:
        sample = sample.split()
        sample = [ps.stem(word) for word in sample]
        sample = ' '.join(sample)
    
    if spell:
        sample = str(TextBlob(text).correct())
    return sample

# %% [code] {"execution":{"iopub.status.busy":"2024-01-30T13:26:03.469187Z","iopub.execute_input":"2024-01-30T13:26:03.469478Z","iopub.status.idle":"2024-01-30T13:26:03.483946Z","shell.execute_reply.started":"2024-01-30T13:26:03.469455Z","shell.execute_reply":"2024-01-30T13:26:03.483018Z"}}
#Checking data

data['correct_text'] = 'text'
data

# %% [code] {"execution":{"iopub.status.busy":"2024-01-30T13:26:03.485254Z","iopub.execute_input":"2024-01-30T13:26:03.485955Z","iopub.status.idle":"2024-01-30T13:26:04.782869Z","shell.execute_reply.started":"2024-01-30T13:26:03.485922Z","shell.execute_reply":"2024-01-30T13:26:04.781761Z"}}
!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/

# %% [code] {"execution":{"iopub.status.busy":"2024-01-30T13:26:04.784574Z","iopub.execute_input":"2024-01-30T13:26:04.785496Z","iopub.status.idle":"2024-01-30T13:26:04.908367Z","shell.execute_reply.started":"2024-01-30T13:26:04.785454Z","shell.execute_reply":"2024-01-30T13:26:04.907539Z"}}
from tqdm import tqdm
tqdm.pandas(desc ="my bar!")
column = 'text'
data[column] = data[column].progress_apply(lambda text: correct_text(str(text)))


filtered_data = data[data['label'] == 0]
g = ' '.join(filtered_data['cleaned_text'])
wordcloud = WordCloud(width = 500, height = 500, background_color = 'black', min_font_size = 10).generate(g)
plt.figure(figsize = (5, 5), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


text_data = data[data['label'] == 0]['cleaned_text'].str.cat(sep=' ')
wordcloud = WordCloud(width = 500, height = 500, background_color = 'black', min_font_size = 10).generate(text_data)
plt.figure(figsize = (5, 5))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

final_data = data.drop(columns = ['text', 'correct_text'])
final_data.to_csv("final_cleaned_data.csv", index = False)
final_data

from gensim.models import FastText
from nltk.tokenize import word_tokenize
from collections import Counter
import warnings

sentences_list = final_data['cleaned_text']
sentences_tokenized = [word_tokenize(sentence.lower()) for sentence in sentences_list]
warnings.filterwarnings('ignore')
fasttext_model = FastText(sentences = sentences_tokenized, min_count = 1)
fasttext_model.train(sentences_tokenized, epochs=25, total_examples=len(sentences_tokenized))
print(fasttext_model)

vocab = Counter()
for sentence in sentences_tokenized:
    vocab.update(sentence)


vocab_file = 'gensim_fasttext_vocab.txt'
lines = [word for word in vocab.keys()]
data = '\n'.join(lines)
with open(vocab_file, 'w', encoding="utf-8") as file:
    file.write(data)

print(vocab.most_common(50))

from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense, Dropout, Embedding, Layer, Concatenate
from keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        # Initialize weight matrices for routing
        self.kernel = self.add_weight(
            shape=[input_shape[-1], self.num_capsule * self.dim_capsule],
            initializer='glorot_uniform',
            trainable=True,
        )

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, -2)  # Expand dims at the correct axis
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsule, 1])  # Tile correctly

        inputs_hat = tf.matmul(inputs_tiled, self.kernel)
        inputs_hat = tf.reshape(inputs_hat, [-1, inputs.shape[1], self.num_capsule, self.dim_capsule])

        b = tf.zeros_like(inputs_hat[:, :, :, 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)  # Softmax along the correct axis
            outputs = self.squash(tf.reduce_sum(c[:, :, :, tf.newaxis] * inputs_hat, axis=2, keepdims=True))
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * outputs, axis=-1)

        return tf.reshape(outputs, [-1, self.num_capsule * self.dim_capsule])

    def squash(self, s):
        squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        scale = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + tf.keras.backend.epsilon())
        return scale * s

# %% [code] {"execution":{"iopub.status.busy":"2024-01-30T13:26:34.830948Z","iopub.execute_input":"2024-01-30T13:26:34.831202Z","iopub.status.idle":"2024-01-30T13:26:34.843163Z","shell.execute_reply.started":"2024-01-30T13:26:34.831169Z","shell.execute_reply":"2024-01-30T13:26:34.84224Z"}}
from keras.models import Sequential
from keras.layers import Bidirectional, GRU, MultiHeadAttention, Dense, Dropout, Reshape, Input
from keras.models import Model  # Importing Model instead of Sequential


def create_model(hidden_units, num_classes): 
    input_layer = Input(shape=(32,))
    reshaped_input = Reshape((1, 32))(input_layer) 

    # Bidirectional GRU Layers
    bidir_gru = Bidirectional(GRU(hidden_units, return_sequences=True))(reshaped_input)
    query = Bidirectional(GRU(hidden_units, return_sequences=True))(bidir_gru)
    key = Bidirectional(GRU(hidden_units, return_sequences=True))(bidir_gru)
    value = Bidirectional(GRU(hidden_units, return_sequences=True))(bidir_gru)

    # MultiHeadAttention Layer
    attended_output = MultiHeadAttention(num_heads=8, key_dim=hidden_units)(query, key, value)
    attended_output = Dense(hidden_units, activation='relu')(attended_output)

    # Capsule Layer
    capsule_layer = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, routings=3)(attended_output)

    # Dense Layers
    dropout_1 = Dropout(0.5)(capsule_layer)
    dense_1 = Dense(hidden_units, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(hidden_units, activation='relu')(dropout_2)
    dropout_3 = Dropout(0.5)(dense_2)
    output = Dense(1, activation='sigmoid')(dropout_3)

    model = Model(inputs=input_layer, outputs=output)
    return model

X = final_data['cleaned_text']
y = final_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.28, random_state = 42)

X_train.fillna(" ", axis = 0, inplace = True)
X_test.fillna(" ", axis = 0, inplace = True)
y_train.fillna(0.0, inplace = True)
y_test.fillna(1.0, inplace = True)

from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer

tokenizer = create_tokenizer(X_train.values)

vocab_set = set([x for x in vocab if len(x) > 2])
tokenizer = create_tokenizer(X_train.values)
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(s.split()) for s in X_train])

from keras.preprocessing.sequence import pad_sequences

def encode_docs(tokenizer, max_length, docs):
    encoded_docs = tokenizer.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs

from keras.utils import pad_sequences
x_train = encode_docs(tokenizer, max_length, X_train.values)
x_test = encode_docs(tokenizer, max_length, X_test.values)

print(x_train.shape)  
print(y_train.shape) 

print(x_test.shape)  
print(y_test.shape) 

hidden_units = 64  
model = create_model(hidden_units, x_train.shape[1])

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])

epochs = 30
batch_size = 32

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test))

model.evaluate(x_test,y_test)

from sklearn.metrics import confusion_matrix, classification_report

Y_pred = model.predict(x_test)
Y_pred = (Y_pred > 0.5)

print(confusion_matrix(y_test, Y_pred))
print(classification_report(y_test, Y_pred))

import numpy as np
input_to_predict = x_test[345]  
input_to_predict = np.reshape(input_to_predict, (1, -1))  
predicted_class = model.predict(input_to_predict)
print(predicted_class[0])