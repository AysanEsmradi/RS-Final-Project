import pandas as pd


def prepare_data():
    '''
    data used is from https://www.kaggle.com/code/nayakroshan/book-recommendation-als-explicit-feedback/data
    excel filename: Preprocessed_data.csv
    '''

    # edit your data path here
    df = pd.read_csv('Book reviews/Preprocessed_data.csv')
    # edit your data path here

    df = df.query('Language=="en"')  # we only recommend english books
    df = df[['user_id', 'isbn', 'rating', 'book_title', 'Summary', 'Category', 'img_m']]
    item_df = df[['isbn', 'book_title', 'Summary', 'Category', 'img_m']].copy()
    item_df.drop_duplicates(subset=['isbn'], inplace=True)
    item_df = item_df.reset_index(drop=True)
    return df, item_df
