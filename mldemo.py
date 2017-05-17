from pandas import read_csv
from nltk.corpus import stopwords, gazetteers
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from string import punctuation
from itertools import groupby
import enchant

date_format = ['MDY', 'DMY']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
prep_file = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\demo code\\prepositions.txt'
prepositions = []
with open(prep_file, 'r') as f:
    for line in f:
        prepositions.append(line.strip().lower())

std_path = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\demo code\\standard-label-simple.csv'
std = read_csv(std_path, header=0, names=['label', 'text'])
stop_words = set(stopwords.words("english"))
filtered_text = []
for text in std.text:
    word = word_tokenize(text)
    filtered_word = [w for w in word if w not in stop_words]
    filtered_text.append(' '.join(filtered_word))
vector = CountVectorizer()
std_dtm = vector.fit_transform(filtered_text)
filename = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\csv files\\Book6.csv'

with open(filename, 'r') as f:
    for line in f:
        string = line.strip().replace('(', '').replace(')', '').replace('"', '')
        string = string.replace(',,', '.').replace(',', ' ').split('.')
        for substring in string:
            input_dtm = vector.transform([substring.lower()])
            if norm(input_dtm.toarray()) == 0:
                continue
            cos_dist = []
            for i in range(len(std_dtm.toarray())):
                cos_dist.append(1 - cosine(std_dtm[i].toarray(), input_dtm.toarray()))
            label_class = []
            for i in range(len(cos_dist)):
                if cos_dist[i] == max(cos_dist):
                    label_class.append(std.label[i])
            s = ''.join(c for c in substring if (ord(c) < 128))
            for c in s:
                if c in set(punctuation) and c not in ['-']:
                    s = s.replace(c, " ")
            for i in range(len(label_class)):
                if label_class[i] == 'code share indicator':
                    for w in s.lower().split():
                        if w in ['no', 'not']:
                            print(label_class[i] + ': ' + 'no')
                            break
                if label_class[i] == 'commission':
                    for (w, t) in pos_tag(word_tokenize(s)):
                        if t == 'CD':
                            print(label_class[i] + ': ' + w + '%')
                            break
                if label_class[i] == 'infant commission':
                    for w in s.lower().split():
                        if w in ['no', 'not']:
                            print(label_class[i] + ': ' + 'no')
                            break
                    for (w, t) in pos_tag(word_tokenize(s)):
                        if t == 'CD':
                            print(label_class[i] + ': ' + w + '%')
                            break
                if label_class[i] == 'sale restriction':
                    for w in s.split():
                        if w in gazetteers.words('countries.txt'):
                            print(label_class[i] + ': ' + w)
                            break
                if label_class[i] == 'tour code':
                    for j in range(len(s.split())):
                        if s.lower().split()[j] == 'code':
                            w = s.split()[j+1]
                            if not enchant.Dict("en_US").check(w):
                                print(label_class[i] + ': ' + w)
                                break
                if label_class[i] in ['ticketing period', 'travelling period']:
                    # print(s)
                    new_w = []
                    for w in s.split():
                        if w.isalnum() and not w.isalpha() and not w.isdecimal():
                            v = []
                            for _, g in groupby(w, str.isalpha):
                                v.append(''.join(list(g)))
                            if v[0] in prepositions:
                                new_w.append(v[0])
                                new_w.append(''.join(v[1:]))
                            else:
                                new_w.append(''.join(v))
                        else:
                            new_w.append(w)
                    new_s = ' '.join(new_w)
                    print(new_s)