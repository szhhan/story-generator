#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:51:17 2020

@author: sizhenhan
"""

import re
import operator
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.python.keras.models import Input, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dropout, Dense,Embedding,SpatialDropout1D,GRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
import copy
from tqdm import tqdm_notebook
import os


class generate_story(object):
    def __init__(self,interval,path,epoch,keep_training=False):
        self.interval = interval
        self.path = path
        self.story_raw, self.sentence_raw, self.text_raw = self.load()
        self.x,self.y, self.x_sentence,self.y_sentence = self.convert(self.story_raw, self.sentence_raw)
        
        self.d, self.wordlist, self.uncommon,self.tokenizer = self.token(self.story_raw)
        
        self.reverse_d, self.embedding, self.uncommon_words = self.embed(self.tokenizer)
        self.X, self.Y = self.final_prepare(self.story_raw,self.tokenizer)
        self.model, self.callbacks_list = self.build_model()
        self.epoch = epoch 
        
        self.path = "model" + str(self.interval) + ".h5"
        if os.path.exists(self.path):
            self.model,self.callbacks_list = self.build_model()
            print(self.model)
            self.model.load_weights(self.path)
        else:
            self.model,self.callbacks_list = self.build_model()
            self.train()
        

        if keep_training:
            self.train()
        
        
    def clean(self,text):
    
        out = copy.deepcopy(text)
        out = out.lower()
        out = out.replace('\d'," ")
        out = out.replace("\t", " ")
        out = out.replace("\n", " ")
        for char in "!#()-./:-_""":
            out = out.replace(char," ")
        out = out.replace("'", "'")
        out = out.replace(',', " , ")
        out = out.replace('.', " . ")
        out = out.replace(';', " ; ")
        out = out.replace('?', " ? ")
        out = out.replace('‘', "'")
        out = out.replace('…', " . ")
        out = out.replace('ç', "c")
        out = out.replace('é', "e")
        out = out.replace('"', " ")
    
        return out
    
    def load(self,p = "Stories/",words_min=10):
        print("loading stories.....")
        print(p)
        files = [i for i in listdir(p) if "txt" in i]
        story_raw = [];
        sentence_raw = [];
        text_raw = "";
        for file in tqdm_notebook(files):
            loc = p + file
            story = open(loc).read()
            story = self.clean(story)
            sentences = re.split("\.|\?", story)
            for sentence in sentences:
                l = sentence.strip().split()
                if len(l) > words_min:
                    sentence_raw.append(sentence)
            story_raw.append(story)
            text_raw += story
        return story_raw, sentence_raw, text_raw
    
    def convert(self,story_raw, sentence_raw):
        x = []
        y = []
        x_sentence = []
        y_sentence = []
        for story in story_raw:
            words = story.split()
            for i in range(0,len(words) - self.interval,self.interval):
                if i + self.interval >= len(words):
                    final = len(words) - 1
                else:
                    final = i + self.interval
                x.append(" ".join(words[i:final]))
                y.append(words[final])
        final = 0 
        for sentence in sentence_raw:
            words = sentence.split()
            words.append(".")
            for i in range(len(words) - self.interval):
                if i + self.interval >= len(words):
                    final = len(words) - 1
                else:
                    final = i + self.interval
                x_sentence.append(" ".join(words[i:final]));
                y_sentence.append(words[final])
        return x,y, x_sentence,y_sentence
    
    def token(self,story_raw,min_appear=2):
        tokenizer = Tokenizer(filters="")
        tokenizer.fit_on_texts(story_raw)
    
        d = tokenizer.word_counts
        d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        wordlist, uncommon = [], []
        for k,v in d:
            if v > min_appear:
                wordlist.append(k);
            else:
                uncommon.append(k);
        return d, wordlist, uncommon,tokenizer
    
    def embed(self,tokenizer):
        embedding_d = {}
        f = open("GloveData/tingle-vectors-300.txt")
        for line in f:
            values = line.split()
            w = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_d[w] = vector
        f.close()

        embedding = np.zeros((len(tokenizer.word_index) + 1, 300))
        uncommon_words = []
        reverse = {}
        for word, ind in tokenizer.word_index.items():
            reverse[ind] = word
            if word in embedding_d:
                embedding[ind] = embedding_d[word]
            else:
                uncommon_words.append(word)
    
        return reverse, embedding, uncommon_words
    
    def final_prepare(self,story_raw,tokenizer):
        all_words= []
        for story in story_raw:
            s = story.split()
            all_words.append(s)
        all_words_flat = [item for sublist in all_words for item in sublist]
    
        X = []
        Y = []
        for i in range(len(all_words_flat)-self.interval):
            X.append(all_words_flat[i:i+self.interval])
            Y.append(all_words_flat[i+self.interval])
    
        X = tokenizer.texts_to_sequences(X)
        Y = tokenizer.texts_to_sequences(Y)
        Y = np_utils.to_categorical(Y, num_classes=len(tokenizer.word_index) + 1)
    
        return X, Y 
    
    def build_model(self):
        input_ = Input(shape=(self.interval,))

        emb = Embedding(len(self.tokenizer.word_index)+1, 300, weights=[self.embedding], trainable=True)(input_)

        lstm_2 = GRU(256)(emb)
        lstm_2 = Dropout(0.2)(lstm_2)

        out = Dense(len(self.tokenizer.word_index)+1, activation='softmax')(lstm_2)
        model = Model(input_, out)
        opt = Adam(lr=0.002)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


        filepath="model" + str(self.interval) + ".h5"
        checkpoint = ModelCheckpoint(filepath, save_weights_only=True),
        callbacks_list = [checkpoint]

        
        return model, callbacks_list
    
    def train(self):
        self.model.fit(self.X,self.Y, batch_size=128,epochs=self.epoch, 
                    callbacks=self.callbacks_list)
    
    def predict(self,sen):
        x = sen
        x2 = self.clean(x)
        x2 = x2.split()
        x2 = self.tokenizer.texts_to_sequences([x2])
        x_final = x2[0]
        x_final = x_final[len(x_final)-7:]
        x_final = np.array(x_final)
        out = ""
        end = 100
        cur = x_final
        i=0

        while i < end:
            cur = cur[1:]
            pred = self.model.predict(cur.reshape((1,6)))
            pred_ind = np.argmax(pred)
            cur = np.append(cur,pred_ind)
            out += self.reverse_d[pred_ind] + " "
            i += 1
        
        return sen + out