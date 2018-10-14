import pandas as pd
import csv
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


ps = PorterStemmer()
emotions = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]
#create sets
emotionset_dict = []
for emotion in emotions:
    new_set = []
    emotionset_dict.append(new_set)

def index_of(emotion):
    return emotions.index(emotion)

#add from word level file
with open('NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as in_file:
    for line in in_file:
        if len(line) != 1:
            line = line.rstrip()
            column_values = line.split("\t")
            if (column_values[2] == '1'):
                word = ps.stem(column_values[0])
                emotionset_dict[index_of(column_values[1])].append(word)
print("first part")
#add from sense level file -> synonyms
with open('NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Senselevel-v0.92.txt', 'r') as in_file:
    i = 0
    for line in in_file:
        if len(line) != 1:
            line = line.rstrip()
            column_values = line.split("\t")
            if (column_values[2] == '1'):
                words = column_values[0].split(", ")
                for word in words:
                    if (word not in emotionset_dict[index_of(column_values[1])]):
                        i += +1
                    emotionset_dict[index_of(column_values[1])].append(word)
print(i)
for emotion in emotions:
    emotion_dict = emotion + ".csv"
    with open(emotion_dict, 'a', newline='') as csvfile_out:
        writer = csv.writer(csvfile_out)
        writer.writerow(['word'])
        for word in set(emotionset_dict[index_of(emotion)]):
            writer.writerow([word])