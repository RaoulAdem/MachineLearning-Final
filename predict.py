from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Model
import os
import cv2
import random
import glob
import pickle
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

base_model=VGG16(weights="imagenet")
base_model=Model(inputs=base_model.inputs,outputs=base_model.layers[-2].output)
model=load_model('best_model.h5')

with open("features.pkl","rb") as f:
  features=pickle.load(f)

with open('captions_arab.txt',"r",encoding="utf-8") as file:
  next(file)
  captions=file.read()

mapping={}
for line in tqdm(captions.split("\n")):
    tokens=line.split(",")
    if len(tokens)<2:
      continue
    else:
      image_id,image_caption=tokens[0],tokens[1:]
    image_id=image_id.split(".")[0]
    image_caption=" ".join(image_caption)
    if image_id not in mapping:
      mapping[image_id]=[]
    mapping[image_id].append(image_caption)

nltk.download("stopwords")

def preprocessing(mapping):
  for id,captions in mapping.items():
    for i in range(len(captions)):
      caption=captions[i]
      caption=caption.lower().strip()
      pattern=re.compile("\W")
      caption=re.sub(pattern," ",caption)
      caption="startseq " + caption + " endseq"
      captions[i]=caption

preprocessing(mapping)
all_captions=[]
for key in mapping:
  for caption in mapping[key]:
    all_captions.append(caption)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(all_captions)
max_length = max(len(caption.split()) for caption in all_captions)

def ind_to_word(ind,tokenizer):
    for word,index in tokenizer.word_index.items():
        if ind==index:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ind_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def transliterate_arabic_to_english(arabic_text):
    arabic_to_english_mapping = {
        'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 't', 'ج': 'j', 'ح': '7', 'خ': 'kh', 'د': 'd',
        'ذ': 'z', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'd', 'ط': 't',
        'ظ': 'z', 'ع': '3', 'غ': 'gh', 'ف': 'f', 'ق': 'k', 'ك': 'k', 'ل': 'l', 'م': 'm',
        'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y', 'ء': '2', 'آ': '2', 'أ': '2', 'ؤ': '2',
        'إ': '2', 'ئ': '2', 'ى': 'y',
        'ة': 'a', 'ﻻ': 'la', 'ﻵ': 'la', 'ﻹ': 'la', 'ﻷ': 'la'
    }
    english_text = ''
    for char in arabic_text:
        english_text += arabic_to_english_mapping.get(char, char)
    english_text = english_text[8:-6]
    return english_text

def result(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = base_model.predict(image, verbose=0)
    p = predict_caption(model, feature, tokenizer, max_length)
    return transliterate_arabic_to_english(p)