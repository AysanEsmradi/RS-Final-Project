import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nlp_summary.training_and_dumping import get_item_tfidf_vector


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


def get_predicted_result(user_preference_df, item_df):
    # todo: load book_tfidf_vector, feature_list, tfidf_matrix
    # todo: prepare user preference df

    # testing
    print("testing...")
    book_tfidf_vector, tfidf_matrix, feature_list = get_item_tfidf_vector(item_df)
    print("end of testing...")

    user_profile = build_user_profile(user_preference_df, book_tfidf_vector, feature_list)
    rec_result = generate_recommendation_results(user_profile, tfidf_matrix, item_df)
    k_results = get_recommendation_list(rec_result, user_preference_df, 10)
    return k_results
