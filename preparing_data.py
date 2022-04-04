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



def prepare_data_with_genre():

    df = pd.read_csv('Preprocessed_data.csv')
    df = df[['user_id', 'isbn', 'rating', 'book_title', 'img_l', 'Summary', 'Category']]
    df.drop(index=df[df['Category'] == '9'].index, inplace=True)    #The 9 is null values in category
    df.drop(index=df[df['Summary'] == '9'].index, inplace=True)    #The 9 is null values in Summary
    print(df.isna().sum())  # check null values

    item_df = df[['isbn', 'book_title', 'Summary', 'Category', 'img_l']].copy()
    item_df.drop_duplicates(subset=['isbn'], inplace=True)
    item_df = item_df.reset_index(drop=True)
    
    item_df['Category'].replace("['Fiction']", 'Fiction',inplace=True)
    item_df['Category'].replace("['Biography']", 'Biography',inplace=True)
    item_df['Category'].replace("['Humor']", 'Humor',inplace=True)
    item_df['Category'].replace("['History']", 'History',inplace=True)
    item_df['Category'].replace("['Religion']", 'Religion',inplace=True)
    item_df['Category'].replace("['Medical']", 'Medical',inplace=True)
    item_df['Category'].replace("['Design']", 'Design',inplace=True)
    
    df_fiction=item_df[item_df['Category']=='Fiction']
    df_bio=item_df[item_df['Category']=='Biography']
    df_Humor=item_df[item_df['Category']=='Humor']
    df_Histo=item_df[item_df['Category']=='History']
    df_reli=item_df[item_df['Category']=='Religion']
    df_medi=item_df[item_df['Category']=='Medical']
    df_design=item_df[item_df['Category']=='Design']
    
    item_df.drop(index=item_df[item_df['Category'] == 'Fiction'].index, inplace=True)
    item_df.drop(index=item_df[item_df['Category'] == 'Biography'].index, inplace=True)
    item_df.drop(index=item_df[item_df['Category'] == 'Humor'].index, inplace=True)
    item_df.drop(index=item_df[item_df['Category'] == 'History'].index, inplace=True)
    item_df.drop(index=item_df[item_df['Category'] == 'Religion'].index, inplace=True)
    item_df.drop(index=item_df[item_df['Category'] == 'Medical'].index, inplace=True)
    item_df.drop(index=item_df[item_df['Category'] == 'Design'].index, inplace=True)
    
    item_df['Category'] = 'Others'
    
    frames = [df_fiction , df_bio,df_Humor,df_Histo,df_reli,df_medi,df_design,item_df]
    
    result = pd.concat(frames)
    
    result['Category'] = result.Category.str.split('|')
    
    Books_with_genres = result.copy(deep=True)

    genre_list = [] # store the occurred genres

    for index, row in result.iterrows():
        for genre in row['Category']:
            Books_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)
    
    Books_with_genres = Books_with_genres.fillna(0)
    
    return df, Books_with_genres
