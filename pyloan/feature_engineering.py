import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time


categorical_feature = ['borrower_identifier', 'term', 'purpose', 'borrower_rating', 'employment', 'region',
                       'facebook', 'twitter', 'paypal', 'ebay', 'linkedin', 'currency', 'base_currency', 'location']


def build_categorical_feature(initial_data, categorical_feature=categorical_feature):
    transformed_data = pd.DataFrame()
    labelEncoders = []
    for feature in categorical_feature:
        le = LabelEncoder()
        labelEncoders.append(le)
        encoded_txt_data = le.fit_transform(initial_data[feature].tolist())
        serie = pd.Series(encoded_txt_data, name=feature)
        transformed_data = pd.concat([transformed_data, serie], axis=1)
    other_feature = [column for column in initial_data.columns.tolist() if column not in categorical_feature]
    transformed_data.set_index(initial_data.index)
    transformed_data = pd.concat([transformed_data, initial_data.loc[:, other_feature]], axis=1)
    return labelEncoders, transformed_data

    #print encoders[0].inverse_transform(transformed_data.ix[0]['loan_identifier'])
    #print data.ix[0]['loan_identifier']
    #print encoders[2].inverse_transform(transformed_data.ix[0]['term'])
    #print data.ix[0]['term']

def encode_data(data):
    hotEncoder = OneHotEncoder()
    return hotEncoder.fit_transform(data.as_matrix())

    
def text_transformation(serie):
    tf = CountVectorizer(token_pattern='[a-zA-Z]{3,}',max_df=0.95, min_df=0.002,max_features=2000,stop_words='english')
    articles_words = tf.fit_transform(serie.to_dict().values())
    word_index = tf.get_feature_names()
    K = 20
    lda = LatentDirichletAllocation(n_topics=K, max_iter=10, learning_method='online', learning_offset=10.,
                                    random_state=0, n_jobs=-1)
    t0 = time()
    new_feature = lda.fit_transform(articles_words)
    print("done in %0.3fs." % (time() - t0))
    return new_feature


#if __name__ == '__main__':
#    initial_data = loan_data()
#    data = build_categorical_feature(initial_data, categorical_feature)
