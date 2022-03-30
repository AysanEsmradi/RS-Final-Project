import pandas as pd


def prepare_data():
    '''
    data used is from https://www.kaggle.com/code/nayakroshan/book-recommendation-als-explicit-feedback/data
    excel filename: Preprocessed_data.csv
    '''

    # edit your data path here
    df = pd.read_csv('Book reviews/Preprocessed_data.csv')
    # edit your data path here

    df = df.query('Language=="en"')
    df = df[['user_id', 'isbn', 'rating', 'book_title', 'Summary', 'Category', 'img_m']]
    df.drop(index=df[df['Category'] == '9'].index, inplace=True)    #The 9 is null values in category
    df.drop(index=df[df['Summary'] == '9'].index, inplace=True)    #The 9 is null values in Summary
    print(df.isna().sum())  # check null values

    item_df = df[['isbn', 'book_title', 'Summary', 'Category', 'img_m']].copy()
    item_df.drop_duplicates(subset=['isbn'], inplace=True)
    item_df = item_df.reset_index(drop=True)
    return df, item_df
