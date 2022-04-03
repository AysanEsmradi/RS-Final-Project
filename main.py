from typing import Optional, List
from surprise import dump

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split
from utils import map_genre
import json
from surprise import KNNBasic
from surprise import Dataset

from preparing_data import prepare_data
from nlp_summary.algorithm import get_predicted_result

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

"""
=================== Body =============================
"""


class Book(BaseModel):
    isbn: int
    book_title: str
    img_l: str
    score: int


# == == == == == == == == == API == == == == == == == == == == =

# show four genres
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children"]}


# show all generes
'''
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]}
'''


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


@app.post("/api/recommend")
def get_recommend(books: list):
    print(books)
    iid = str(sorted(books, key=lambda i: i.score, reverse=True)[0].isbn)
    score = int(sorted(books, key=lambda i: i.score, reverse=True)[0].score)
    res = get_initial_items(iid, score)
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    print(res)
    rec_movies = data.loc[data['isbn'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['isbn', 'book_title', 'img_l', 'like']]
    return json.loads(results.to_json(orient="records"))


@app.get("/api/add_recommend/{item_id}")
async def add_recommend(item_id):
    res = get_similar_items(str(item_id), n=5)
    res = [int(i) for i in res]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


def user_add(iid, score):
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    df.to_csv('new_' + 'u.data')
    with open(r'new_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        data_input = []
        s = [user, str(iid), int(score), '0']
        data_input.append(s)
        for k in data_input:
            wf.writerow(k)


def get_initial_items(iid, score, n=12):
    res = []
    user_add(iid, score)
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./model', algo=algo, verbose=1)
    all_results = {}
    for i in range(1682):
        uid = str(944)
        iid = str(i)
        pred = algo.predict(uid, iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res


def get_similar_items(iid, n=12):
    algo = dump.load('./model')[1]
    inner_id = algo.trainset.to_inner_iid(iid)
    print(inner_id)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print(neighbors_iid)
    return neighbors_iid


# == == == == == == == == == API for Jeffy == == == == == == == == == == =

@app.post("/api/recommend_nlp")
def get_recommend_by_nlp(books: list):
    explicit_user_df = pd.DataFrame(books)
    new_user_id = add_new_nlp_user(explicit_user_df)     # add new user
    print(new_user_id)
    explicit_user_df.loc[:, 'user_id'] = new_user_id
    explicit_user_df.loc[:, 'rating'] = explicit_user_df.loc[:, 'score']
    k_results = get_predicted_result(explicit_user_df, item_df)

    print("========================== return nlp result ============================")
    rec_books = item_df.loc[item_df['isbn'].isin(k_results)]
    rec_books.loc[:, 'like'] = None
    k_results = rec_books.loc[:, ['isbn', 'book_title', 'img_l', 'like']]
    print(k_results)

    '''
    iid = str(sorted(books, key=lambda i: i.score, reverse=True)[0].isbn)
    score = int(sorted(books, key=lambda i: i.score, reverse=True)[0].score)
    res = get_initial_items(iid, score)
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    print(res)
    rec_movies = data.loc[data['isbn'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['isbn', 'book_title', 'img_l', 'like']]
    '''

    return json.loads(k_results.to_json(orient="records"))

# todo: like items
# todo: get_similar_items by nlp


def add_new_nlp_user(explicit_user_df):
    nlp_user_file = "./new_nlp_data.csv"
    if os.path.exists(nlp_user_file):
        nlp_new_user_df = pd.read_csv(nlp_user_file)
        user_list = nlp_new_user_df['user_id'].unique()
        user_list = np.sort(user_list)
        new_user_id = user_list[-1] + 1
        explicit_user_df.loc[:, 'user_id'] = new_user_id
        nlp_new_user_df = pd.concat([nlp_new_user_df, explicit_user_df])
        nlp_new_user_df.reset_index(drop=True)
    else:
        nlp_new_user_df = explicit_user_df.copy()
        new_user_id = 1
        nlp_new_user_df.loc[:, 'user_id'] = new_user_id
    nlp_new_user_df.to_csv(nlp_user_file, index=False)
    return new_user_id
