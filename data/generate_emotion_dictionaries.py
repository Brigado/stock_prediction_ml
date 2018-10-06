import pandas as pd
import csv

with open('NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as in_file:
    i = 0
    for line in in_file:
        if len(line) != 1:
            line = line.rstrip()
            column_values = line.split("\t")
            if (column_values[2] == '1'):
                emotion_dict = column_values[1] + '.csv'
                with open(emotion_dict, 'a', newline='') as csvfile_out:
                    writer = csv.writer(csvfile_out)
                    writer.writerow([column_values[0]])