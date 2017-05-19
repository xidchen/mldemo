from pandas import read_csv
from nltk.corpus import stopwords, gazetteers
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from string import punctuation
from itertools import groupby
import enchant

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
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
    word = word_tokenize(text)
    filtered_word = [w for w in word if w not in stop_words]
    filtered_text.append(' '.join(filtered_word))
vector = CountVectorizer()
std_dtm = vector.fit_transform(filtered_text)
filename = 'C:\\Users\\sheldonc\\Documents\\CA CS ml demo\\csv files\\Book7.csv'

with open(filename, 'r') as f:
    for line in f:
        string = line.strip().replace('(', '').replace(')', '').replace('"', '')
        string = string.replace(',,', '.').replace(',', ' ').replace(':', ' ').split('.')
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
                    for (w, sword) in pos_tag(word_tokenize(s)):
                        if sword == 'CD':
                            print(label_class[i] + ': ' + w + '%')
                            break
                if label_class[i] == 'infant commission':
                    for w in s.lower().split():
                        if w in ['no', 'not']:
                            print(label_class[i] + ': ' + 'no')
                            break
                    for (w, sword) in pos_tag(word_tokenize(s)):
                        if sword == 'CD':
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
                    sword = s.split()
                    new_sword = []
                    for j in range(len(sword)):
                        # Process case like "RELEASED: DEC 29, 201514-"
                        if sword[j].lower() == 'released':
                            if sword[j+1].lower() in months or sword[j+2].lower() in months:
                                sword[j+3] = sword[j+3][:4]
                                if sword[j+3].isdecimal():
                                    released_date = ' '.join(sword[j+1:j+4])
                        # Process case like "Ticket must be issued on/before31JAN, 2016"
                        if sword[j].isalnum() and not sword[j].isalpha() and not sword[j].isdecimal():
                            for k, g in groupby(sword[j], str.isalpha):
                                new_sword.append(''.join(list(g)))
                        else:
                            new_sword.append(sword[j])
                    new_s = ' '.join(new_sword)
                    for j in range(len(new_sword)):
                        if new_sword[j] in prepositions and (new_sword[j+1].isdecimal() or new_sword[j+1].lower() in months):
                            # "Starting from Mar27, 2016 to Dec31, 2016"
                            if j+7 in range(len(new_sword)) and new_sword[j+4] in prepositions:
                                if new_sword[j+5].isdecimal() or new_sword[j+5].lower() in months:
                                    u = ' '.join(new_sword[j:j+8])
                                    print(label_class[i] + ': ' + u)
                                    break
                            # Process case like "Ticket must be issued on/before 29FEB, 2016"
                            elif new_sword[j-1] in prepositions:
                                u = ' '.join(new_sword[j-1:j+4])
                                print(label_class[i] + ': ' + u)
                                break
                            # Process case like "Ticketing valid until 18FEB16"
                            else:
                                u = ' '.join(new_sword[j:j+4])
                                print(label_class[i] + ': ' + u)
                                break
                        # Process case like "TICKETING PERIOD:      NOW - FEB 02, 2016"
                        # Process case like "TRAVELING DATES:      NOW - FEB 10,2016    FEB 22,2016 - MAY 12,2016"
                        if new_sword[j] in ['-'] and (new_sword[j+1].lower() in months or new_sword[j+2].lower() in months):
                            if new_sword[j-1].lower() == 'now':
                                u = released_date + ' - ' + ' '.join(new_sword[j+1:j+4])
                                print(label_class[i] + ': ' + u)
                            elif new_sword[j-3].lower() in months or new_sword[j-2].lower() in months:
                                u = ' '.join(new_sword[j-3:j+4])
                                print(label_class[i] + ': ' + u)