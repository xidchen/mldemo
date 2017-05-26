from pandas import read_csv
from nltk.corpus import stopwords, gazetteers
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from itertools import groupby
import enchant
import re

month_file = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\demo code\\month_list.txt'
months = []
with open(month_file, 'r') as f:
    for line in f:
        months.append(line.strip().lower())
prep_file = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\demo code\\prepositions.txt'
prepositions = []
with open(prep_file, 'r') as f:
    for line in f:
        prepositions.append(line.strip().lower())
released_date = ''

std_path = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\demo code\\standard-label-simple.csv'
std = read_csv(std_path, header=0, names=['label', 'text'])
stop_words = set(stopwords.words("english"))
filtered_text = []
for text in std.text:
    filtered_word = [w for w in word_tokenize(text) if w not in stop_words]
    filtered_text.append(' '.join(filtered_word))
vector = CountVectorizer()
std_dtm = vector.fit_transform(filtered_text)
filename = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\larger set\\T23 csv files\\T23-1.csv'

with open(filename, 'r') as f:
    for line in f:
        string = line.strip().replace('(', '').replace(')', '').replace('"', '').replace('*', '')
        string = string.replace('—', ' - ').replace('-', ' - ').replace(',,', '.')
        string = string.replace(',', ' ').replace(':', ' ').replace('/', ' ')
        string = string.split('.')
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
            label_class = list(set(label_class))
            s = ''.join(c for c in substring if (ord(c) < 128))
            for i in range(len(label_class)):
                if label_class[i] == 'code share indicator':
                    for w in s.lower().split():
                        if w in ['no', 'not']:
                            print(label_class[i] + ': ' + 'no')
                if label_class[i] in ['commission', 'infant commission']:
                    for w in s.lower().split():
                        if w in ['no', 'not']:
                            print(label_class[i] + ': ' + 'no')
                    match = re.search('(\d+%)', s)
                    if match:
                        pct = match.group(1)
                        print(label_class[i] + ': ' + pct)
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
                if label_class[i] in ['ticketing period', 'travelling period']:
                    w = s.split()
                    nw = []
                    for j in range(len(w)):
                        # Process case like "RELEASED: DEC 29, 201514-"
                        if w[j].lower() == 'released':
                            if w[j+1].lower() in months or w[j+2].lower() in months:
                                w[j+3] = w[j+3][:4]
                                if w[j+3].isdecimal():
                                    released_date = ' '.join(w[j+1:j+4])
                        # Process case like "Ticket must be issued on/before31JAN, 2016"
                        if w[j].isalnum() and not w[j].isalpha() and not w[j].isdecimal():
                            for k, g in groupby(w[j], str.isalpha):
                                nw.append(''.join(list(g)))
                        else:
                            nw.append(w[j])
                    for j in range(len(nw)):
                        if j+2 in range(len(nw)) and nw[j] in prepositions:
                            if nw[j+1].lower() in months or nw[j+2].lower() in months:
                                # Process case like "Starting from Mar27, 2016 to Dec31, 2016"
                                if j+7 in range(len(nw)) and nw[j+4] in prepositions:
                                    if nw[j+5].lower() in months or nw[j+6].lower() in months:
                                        u = ' '.join(nw[j:j+8])
                                        print(label_class[i] + ': ' + u)
                                # Process case like "Ticket must be issued on/before 29FEB, 2016"
                                elif nw[j-1] in prepositions:
                                    u = ' '.join(nw[j-1:j+4])
                                    print(label_class[i] + ': ' + u)
                                # Process case like "Ticketing valid until 18FEB16"
                                else:
                                    u = ' '.join(nw[j:j+4])
                                    print(label_class[i] + ': ' + u)
                        # Process case like "TICKETING PERIOD:      NOW - FEB 02, 2016"
                        # Process case like "TRAVELING DATES:      NOW - FEB 10,2016    FEB 22,2016 - MAY 12,2016"
                        if j+2 in range(len(nw)) and nw[j] in ['-']:
                            if nw[j+1].lower() in months or nw[j+2].lower() in months:
                                if nw[j-1].lower() == 'now':
                                    u = released_date + ' - ' + ' '.join(nw[j+1:j+4])
                                    print(label_class[i] + ': ' + u)
                                elif nw[j-3].lower() in months or nw[j-2].lower() in months:
                                    u = ' '.join(nw[j-3:j+4])
                                    print(label_class[i] + ': ' + u)