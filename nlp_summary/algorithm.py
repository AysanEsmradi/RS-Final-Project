import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profile(user_preference_df, book_tfidf_vector, feature_list, normalized=False):
    user_preference_df = user_preference_df[['user_id', 'isbn', 'book_title', 'rating']]
    user_preference_df.reset_index(drop=True)

    # merge TFIDF vector to user preference df
    user_book_rating_df = pd.merge(user_preference_df, book_tfidf_vector)
    if user_preference_df.rating.sum() != 0:
        rating_weight = user_preference_df.rating / user_preference_df.rating.sum()
    else:
        rating_weight = user_preference_df.rating
    user_profile = user_book_rating_df[feature_list].T.dot(rating_weight)
    if normalized:
        user_profile = user_profile / sum(user_profile.values)
    return user_profile


def generate_recommendation_results(user_profile, tfidf_matrix, item_df):
    # predicting using cosine similarity
    u_v = user_profile.values
    u_v_matrix = [u_v]
    recommendation_table = cosine_similarity(u_v_matrix, tfidf_matrix)
    recommendation_table_df = item_df[['isbn', 'book_title']].copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)
    return result


def get_recommendation_list(rec_result, user_preference_df, k):
    top_k_recommendation = []
    item_in_user_pre = user_preference_df.isbn.unique()
    for rec_isbn in rec_result.isbn.values:
        if rec_isbn not in item_in_user_pre:
            top_k_recommendation.append(rec_isbn)
        if len(top_k_recommendation) == k:
            break
    return top_k_recommendation


def get_predicted_nlp_result(user_preference_df, item_df, k):
    # todo: prepare user preference df

    # testing
    print("loading tfidf item...")
    book_tfidf_vector = pd.read_csv("nlp_summary/tfidf_vector.csv")
    tfidf_matrix = scipy.sparse.load_npz('nlp_summary/tfidf_matrix.npz')
    feature_list = np.load('nlp_summary/feature_list.npy', allow_pickle=True)
    print("end of loading tfidf item...")

    user_profile = build_user_profile(user_preference_df, book_tfidf_vector, feature_list)
    rec_result = generate_recommendation_results(user_profile, tfidf_matrix, item_df)
    k_results = get_recommendation_list(rec_result, user_preference_df, k)
    return k_results
