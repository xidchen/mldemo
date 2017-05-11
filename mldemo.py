from pandas import read_csv
from nltk.corpus import stopwords, gazetteers
from nltk import word_tokenize, ne_chunk, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from string import punctuation
import enchant

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
filename = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\csv files\\Book1.csv'
with open(filename, 'r') as f:
    for line in f:
        string = line.strip().replace('(', '').replace(')', '').replace('"', '').split('.')
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
            print(substring)
            # print(label_class, max(cos_dist))
            # print(ne_chunk(pos_tag(word_tokenize(substring))))
            s = ''.join(c for c in substring if (ord(c) < 128))
            for c in s:
                if c in set(punctuation):
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
