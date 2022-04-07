from surprise import dump
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os
import json
from surprise import SVD, Reader, Dataset
from preparing_data import prepare_data
from nlp_summary.algorithm import get_predicted_nlp_result

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================
data, item_df = prepare_data()
print(data.columns)
nlp_user_file = "./new_nlp_data.csv"
svd_user_file = "./new_svd_data.csv"

"""
=================== Body =============================
"""


class Book(BaseModel):
    book_title: str
    feedback: int
    img_l: str
    isbn: str


# == == == == == == == == == API == == == == == == == == == == =

# show four genres
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children"]}


@app.post("/api/books")
def get_movies(genre: list):
    # todo: select algorithm
    '''
    # use of genre

    print(genre)
    query_str = " or ".join(map(map_genre, genre))
    results = data.query(query_str)
    '''

    results = data
    results.loc[:, 'score'] = None
    results = results.sample(18).loc[:, ['isbn', 'book_title', 'img_l', 'score']]
    return json.loads(results.to_json(orient="records"))


# == == == == == == == == == API for NLP == == == == == == == == == == =

@app.post("/api/recommend_nlp")
def get_recommend_nlp(books: list):
    explicit_user_df = pd.DataFrame(books)
    new_user_id = add_new_nlp_user(explicit_user_df)  # add new user
    explicit_user_df.loc[:, 'user_id'] = new_user_id
    explicit_user_df.loc[:, 'rating'] = explicit_user_df.loc[:, 'score']
    k_results = get_predicted_nlp_result(explicit_user_df, item_df, k=20)

    print("========================== return nlp result ============================")
    rec_books = item_df.loc[item_df['isbn'].isin(k_results)]
    rec_books.loc[:, 'score'] = None
    k_results = rec_books.loc[:, ['isbn', 'book_title', 'img_l', 'score']]
    return json.loads(k_results.to_json(orient="records"))


@app.post("/api/add_recommend_nlp")
async def add_recommend_nlp(feedback: list):
    feedback_df = pd.DataFrame(feedback)
    nlp_new_user_df = load_new_nlp_user(feedback_df)
    nlp_new_user_df.loc[:, 'rating'] = nlp_new_user_df.loc[:, 'score']
    k_results = get_predicted_nlp_result(nlp_new_user_df, item_df, k=5)
    rec_books = item_df.loc[item_df['isbn'].isin(k_results)]
    rec_books.loc[:, 'score'] = None
    k_results = rec_books.loc[:, ['isbn', 'book_title', 'img_l', 'score']]
    return json.loads(k_results.to_json(orient="records"))


@app.get("/stage_two", response_class=FileResponse)
def commence_stage_two():
    return FileResponse('stage_2.html')


def add_new_nlp_user(explicit_user_df):
    user_file = nlp_user_file

    if os.path.exists(user_file):
        new_user_df = pd.read_csv(user_file)
        user_list = new_user_df['user_id'].unique()
        user_list = np.sort(user_list)
        new_user_id = user_list[-1] + 1
        explicit_user_df.loc[:, 'user_id'] = new_user_id
        new_user_df = pd.concat([new_user_df, explicit_user_df])
        new_user_df.reset_index(drop=True, inplace=True)
    else:
        new_user_df = explicit_user_df.copy()
        new_user_id = 1
        new_user_df.loc[:, 'user_id'] = new_user_id

    new_user_df.to_csv(user_file, index=False)
    return new_user_id


def load_new_nlp_user(feedback_df):
    nlp_new_user_df = pd.read_csv(nlp_user_file)
    user_list = nlp_new_user_df['user_id'].unique()
    user_list = np.sort(user_list)
    new_user_id = user_list[-1]
    feedback_df.loc[:, 'user_id'] = new_user_id
    nlp_new_user_df.reset_index(drop=True, inplace=True)
    nlp_new_user_df.to_csv(svd_user_file, index=False)

    nlp_new_user_df = pd.concat([nlp_new_user_df, feedback_df])
    nlp_new_user_df = nlp_new_user_df.loc[nlp_new_user_df['user_id'].isin([new_user_id])]
    nlp_new_user_df.reset_index(drop=True, inplace=True)
    return nlp_new_user_df


# == == == == == == == == == API for SVD == == == == == == == == == == =

@app.post("/api/recommend_svd")
def get_recommend_svd(books: list):
    explicit_user_df = pd.DataFrame(books)

    new_user_id = add_new_svd_user(explicit_user_df)  # add new user
    explicit_user_df.loc[:, 'user_id'] = new_user_id
    explicit_user_df.loc[:, 'rating'] = explicit_user_df.loc[:, 'score']

    data_with_new_user = pd.concat(
        [data[['user_id', 'isbn', 'rating']], explicit_user_df[['user_id', 'isbn', 'rating']]])
    data_with_new_user.reset_index(drop=True, inplace=True)

    train_set, algo_svd = train_and_dump_svd(data_with_new_user)
    k_results = get_svd_predicted_results(algo_svd, new_user_id)

    print("========================== return svd result ============================")
    rec_books = item_df.loc[item_df['isbn'].isin(k_results)]
    rec_books.loc[:, 'score'] = None
    k_results = rec_books.loc[:, ['isbn', 'book_title', 'img_l', 'score']]
    return json.loads(k_results.to_json(orient="records"))


@app.post("/api/add_recommend_svd")
async def add_recommend_svd(feedback: list):
    feedback_df = pd.DataFrame(feedback)
    svd_new_user_df, new_user_id = load_new_svd_user(feedback_df)
    svd_new_user_df.loc[:, 'rating'] = svd_new_user_df.loc[:, 'score']

    data_with_feedback = pd.concat(
        [data[['user_id', 'isbn', 'rating']], svd_new_user_df[['user_id', 'isbn', 'rating']]])
    data_with_feedback.reset_index(drop=True, inplace=True)

    train_set, algo_svd = train_and_dump_svd(data_with_feedback, is_dump=False)
    k_results = get_svd_predicted_results(algo_svd, new_user_id, k=5)

    print("========================== return svd feedback ============================")
    rec_books = item_df.loc[item_df['isbn'].isin(k_results)]
    rec_books.loc[:, 'score'] = None
    k_results = rec_books.loc[:, ['isbn', 'book_title', 'img_l', 'score']]

    return json.loads(k_results.to_json(orient="records"))


def add_new_svd_user(explicit_user_df):
    user_file = svd_user_file

    if os.path.exists(user_file):
        new_user_df = pd.read_csv(user_file)
        user_list = new_user_df['user_id'].unique()
        user_list = np.sort(user_list)
        new_user_id = user_list[-1] + 1
        explicit_user_df.loc[:, 'user_id'] = new_user_id
        new_user_df = pd.concat([new_user_df, explicit_user_df])
        new_user_df.reset_index(drop=True, inplace=True)
    else:
        # create new user id
        user_id_list = data['user_id'].unique()
        user_id_list = np.sort(user_id_list)
        new_user_id = user_id_list[-1] + 1

        new_user_df = explicit_user_df.copy()
        new_user_df.loc[:, 'user_id'] = new_user_id

    new_user_df.to_csv(user_file, index=False)
    return new_user_id


def train_and_dump_svd(train_df, is_dump=True):
    reader = Reader(rating_scale=(0, 10))
    train_raw_data = Dataset.load_from_df(train_df[['user_id', 'isbn', 'rating']], reader)
    train_set = train_raw_data.build_full_trainset()

    # for hyper-parameter tuning, please refer to the jupyter notebook
    algo_svd = SVD(n_factors=10, n_epochs=20, biased=False)
    print("training SVD...")
    algo_svd.fit(train_set)
    if is_dump:
        dump.dump('./svd.model', algo=algo_svd, verbose=1)
    return train_set, algo_svd


def get_svd_predicted_results(algo_svd, new_user_id, k=20):
    res = []
    all_results = {}
    all_isbn = item_df.isbn.unique()

    print("svd predicting...")
    for isbn in all_isbn:
        pred = algo_svd.predict(new_user_id, isbn).est
        all_results[isbn] = pred
    sorted_list = sorted(all_results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(k):
        res.append(sorted_list[i][0])
    return res


def load_new_svd_user(feedback_df):
    svd_new_user_df = pd.read_csv(svd_user_file)
    user_list = svd_new_user_df['user_id'].unique()
    user_list = np.sort(user_list)
    new_user_id = user_list[-1]
    feedback_df.loc[:, 'user_id'] = new_user_id
    svd_new_user_df = pd.concat([svd_new_user_df, feedback_df])
    svd_new_user_df.reset_index(drop=True, inplace=True)
    svd_new_user_df.to_csv(svd_user_file, index=False)

    svd_new_user_df = svd_new_user_df.loc[svd_new_user_df['user_id'].isin([new_user_id])]
    svd_new_user_df.reset_index(drop=True, inplace=True)
    return svd_new_user_df, new_user_id
