import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_summary.processing import preprocessing
from preparing_data import prepare_data


def get_item_tfidf_vector(item_df, max_feat=100):
    # converting summary to TFIDF matrix
    tfidf = TfidfVectorizer(
        # preprocessor=preprocessing,
        ngram_range=(1, 1),  # todo: tune hyperparameter
        max_features=max_feat
    )

    tfidf_matrix = tfidf.fit_transform(item_df['Summary'])
    book_tfidf_vector = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    feature_list = tfidf.get_feature_names_out()
    book_tfidf_vector['isbn'] = item_df['isbn']
    return book_tfidf_vector, tfidf_matrix, feature_list
