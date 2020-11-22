import nltk
import numpy as np
import pprint
import treetaggerwrapper
#nltk.download('punkt') #---> Run this only the first time
#nltk.download('wordnet') #---> Run this only the first time
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

tagger = treetaggerwrapper.TreeTagger(TAGLANG='it')
wordnet_lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("italian")

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def lem(word):
    return wordnet_lemmatizer.lemmatize(word, pos='v')

def lemmatize(all_words):
    tags = tagger.tag_text(all_words)
    tags2 = treetaggerwrapper.make_tags(tags)
    return tags2

def bag_of_words(tekenized_sentence, all_words):
    #Lemmatize
    
    t_lem = ''
    for w in tekenized_sentence:
            t_lem = t_lem + ' ' + w

    lemma = lemmatize(t_lem)
    all_words_lemma = []
    for w in lemma:
        all_words_lemma.append(w[2])
        
    tokenize_sentence = [stem(w) for w in all_words_lemma]
    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0

    return bag
