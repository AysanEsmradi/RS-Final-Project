import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_summary.processing import preprocessing
from preparing_data import prepare_data


def get_item_tfidf_vector(item_df, max_feat=200):
    # converting summary to TFIDF matrix
    tfidf = TfidfVectorizer(
        preprocessor=preprocessing,
        ngram_range=(1, 1),  # todo: tune hyperparameter
        max_features=max_feat
    )

    tfidf_matrix = tfidf.fit_transform(item_df['Summary'])
    book_tfidf_vector = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    feature_list = tfidf.get_feature_names_out()
    book_tfidf_vector['isbn'] = item_df['isbn']
    return book_tfidf_vector, tfidf_matrix, feature_list


def main():
    _, item_df = prepare_data()
    book_tfidf_vector, tfidf_matrix, feature_list = get_item_tfidf_vector(item_df, 200)
    book_tfidf_vector.to_csv("tfidf_vector.csv")
    scipy.sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)
    np.save("feature_list.npy", feature_list)


if __name__ == "__main__":
    main()
