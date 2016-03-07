import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time


categorical_feature = ['borrower_identifier', 'term', 'purpose', 'borrower_rating', 'employment', 'region',
                       'facebook', 'twitter', 'paypal', 'ebay', 'linkedin', 'currency', 'base_currency', 'location']


def re_assemble_dataset(initial_data, dataframe, feature_to_join):
    dataframe.set_index(initial_data.index)
    other_feature = [column for column in initial_data.columns.tolist() if column not in feature_to_join]
    return pd.concat([dataframe, initial_data.loc[:, other_feature]], axis=1)

def build_categorical_feature(initial_data, categorical_feature=categorical_feature, binary_encoding=False):
    transformed_data = pd.DataFrame()
    labelEncoders = []
    for feature in categorical_feature:
        le = LabelEncoder()
        labelEncoders.append(le)
        encoded_txt_data = le.fit_transform(initial_data[feature].tolist())
        serie = pd.Series(encoded_txt_data, name=feature)
        transformed_data = pd.concat([transformed_data, serie], axis=1)
    if binary_encoding:
        enc_data = encode_data(transformed_data).toarray()
        transformed_data = pd.DataFrame(enc_data)
#    mapper = {0:'term', 1:'term', 2:'term', 3:'term',
#          4:'purpose', 5:'purpose', 6:'purpose', 7:'purpose', 8:'purpose', 9:'purpose', 10:'purpose',
#          11:'employment', 12:'employment', 13:'employment', 14:'employment', 15:'employment'}
#    encoded_data.rename_axis(mapper,axis=1,inplace=True)
    return labelEncoders, re_assemble_dataset(initial_data, transformed_data, categorical_feature)

    #print encoders[0].inverse_transform(transformed_data.ix[0]['loan_identifier'])
    #print data.ix[0]['loan_identifier']
    #print encoders[2].inverse_transform(transformed_data.ix[0]['term'])
    #print data.ix[0]['term']

def encode_data(data):
    hotEncoder = OneHotEncoder()
    return hotEncoder.fit_transform(data.as_matrix())

    
def text_transformation(initial_data, y, categorical_feature=['project_description']):
    tf = CountVectorizer(token_pattern='[a-zA-Z]{3,}',max_df=0.95, min_df=0.002,
                         max_features=2000, stop_words='english')
    serie = initial_data[categorical_feature[0]]
    articles_words = tf.fit_transform(serie.to_dict().values(), y)
    word_index = tf.get_feature_names()
    K = 20
    lda = LatentDirichletAllocation(n_topics=K, max_iter=10, learning_method='online', learning_offset=10.,
                                    random_state=0, n_jobs=-1)
    t0 = time()
    new_feature = lda.fit_transform(articles_words)
    print("done in %0.3fs." % (time() - t0))
    new_feature = pd.DataFrame(new_feature)
    return re_assemble_dataset(initial_data, new_feature, categorical_feature), lda


#if __name__ == '__main__':
#    initial_data = loan_data()
#    data = build_categorical_feature(initial_data, categorical_feature)
