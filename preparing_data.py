import pandas as pd


def prepare_data(path='./Preprocessed_data.csv'):
    '''
    data used is from https://www.kaggle.com/code/nayakroshan/book-recommendation-als-explicit-feedback/data
    excel filename: Preprocessed_data.csv
    '''

    # edit your data path here
    df = pd.read_csv(path)
    # edit your data path here

    df = df.query('Language=="en"')
    df = df[['user_id', 'isbn', 'rating', 'book_title', 'Summary', 'Category', 'img_l']]
    df.drop(index=df[df['Category'] == '9'].index, inplace=True)    #The 9 is null values in category
    df.drop(index=df[df['Summary'] == '9'].index, inplace=True)    #The 9 is null values in Summary

    item_df = df[['isbn', 'book_title', 'Summary', 'Category', 'img_l']].copy()
    item_df.drop_duplicates(subset=['isbn'], inplace=True)
    item_df.drop_duplicates(subset=['book_title'], inplace=True)
    item_df.reset_index(drop=True, inplace=True)

    df = df.loc[df['isbn'].isin(item_df.isbn)]
    df.reset_index(drop=True, inplace=True)
    return df, item_df
