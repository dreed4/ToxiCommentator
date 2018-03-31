import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
subm = pd.read_csv('sample_submission.csv')

#Splits comments into word units
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#Create a new column for those entries that are not toxic
train['none'] = 1-train[label_cols].max(axis=1)

#Fill empty cells
train['comment_text'].fillna("empty_comment", inplace=True)
test['comment_text'].fillna("empty_comment", inplace=True)

#regular expression that contains punctuation and other ascii characters
re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')

n = train.shape[0]

vec = TfidfVectorizer(analyzer='word', tokenizer=tokenize, min_df=3, max_df=.9, strip_accents='unicode', \
                      use_idf=True, smooth_idf=True, sublinear_tf=True)
'''
#analyzer: Builds features from words, instead of characters
#tokenizer: Uses tokenize function to break comments into words
#min_df: If a word does not appear more than 3 times, it is ignored
#max_df: If a word appears in 90% of documents, or more, ignore it
#strip_accents: used for preprocessing
#use_idf: Inverse Document Frequency Reweighing (slide)
#sublinear_tf: Regularization, creates a new document that contains every word exactly once
'''

#Learns the vocabulary and inverse document frequency, gives us a term-document matrix:
#Inverse document frequency is used to supress the effect of common words like 'a', 'and', and 'the'
training_term_doc = vec.fit_transform(train['comment_text'])

#Gives us a term-document matrix to test vocabulary on:
test_term_doc = vec.transform(test['comment_text'])

#Returns Naive Bayes feature probability
def prob(y_i, y):
    p = training_term_doc[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def model(column):
    '''This function takes as parameter an entire class column from the dataset,
calculates the probability of a positive (1) and negative (0) label for that feature,
and fits a logistic model between the training term-document-matrix and the associated
probability of having that feature.'''
    column = column.values
    r = np.log(prob(1,column) / prob(0,column))
    m = LogisticRegression(C=1, dual=True)
    x_nb = training_term_doc.multiply(r)
    return m.fit(x_nb, column), r

preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print(j)
    m,r = model(train[j])
    preds[:,i] = m.predict_proba(test_term_doc.multiply(r))[:,1]

pred_ids = pd.DataFrame({'id': subm["id"]})
pred_toxic = pd.concat([pred_ids, pd.DataFrame(preds, columns = label_cols)], axis=1)
pred_toxic.to_csv('predictions.csv', index=False)
