import os
import re
import pandas as pd
import json
import glob

from typing import List, Tuple
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# CHANGE THIS TO YOUR OPENAI API KEY

os.environ["OPENAI_API_KEY"] = ""
CHAT = ChatOpenAI(model="gpt-4-32k",temperature=0,max_tokens=2000)
dev_db_path = "dev/dev_databases"
dev_df = pd.read_json("dev/dev.json")

# ----------------------- #

SYSTEM_SCHEMA_LINKING_TEMPLATE = """
You are an agent designed to find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.
Hint helps you to fine the correct schema_links.
###
Few examples of this task are:
###
Schema of the database with sample rows and column descriptions:
#
CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)

/*
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/

CREATE TABLE ratings (
        movie_id INTEGER, 
        rating_id INTEGER, 
        rating_url TEXT, 
        rating_score INTEGER, 
        rating_timestamp_utc TEXT, 
        critic TEXT, 
        critic_likes INTEGER, 
        critic_comments INTEGER, 
        user_id INTEGER, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_eligible_for_trial INTEGER, 
        user_has_payment_method INTEGER, 
        FOREIGN KEY(movie_id) REFERENCES movies (movie_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id), 
        FOREIGN KEY(rating_id) REFERENCES ratings (rating_id), 
        FOREIGN KEY(user_id) REFERENCES ratings_users (user_id)
)

/*
3 rows from ratings table:
movie_id        rating_id       rating_url      rating_score    rating_timestamp_utc    critic  critic_likes    critic_comments user_id user_trialist   user_subscriber       user_eligible_for_trial user_has_payment_method
1066    15610495        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/15610495 3       2017-06-10 12:38:33     None    0       0       41579158     00       1       0
1066    10704606        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10704606 2       2014-08-15 23:42:31     None    0       0       85981819     11       0       1
1066    10177114        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10177114 2       2014-01-30 13:21:57     None    0       0       4208563 0    01       1
*/

Table: movies
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi

Table: ratings
Column movie_id: column description -> Movie ID related to the rating
Column rating_id: column description -> Rating ID on Mubi
Column rating_url: column description -> URL to the rating on Mubi
Column rating_score: column description -> Rating score ranging from 1 (lowest) to 5 (highest), value description -> commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie
Column rating_timestamp_utc : column description -> Timestamp for the movie rating made by the user on Mubi
Column critic: column description -> Critic made by the user rating the movie. , value description -> If value = "None", the user did not write a critic when rating the movie.
Column critic_likes: column description -> Number of likes related to the critic made by the user rating the movie
Column critic_comments: column description -> Number of comments related to the critic made by the user rating the movie
Column user_id: column description -> ID related to the user rating the movie
Column user_trialist : column description -> whether user was a tralist when he rated the movie, value description -> 1 = the user was a trialist when he rated the movie 0 = the user was not a trialist when he rated the movie
#
Q: Which year has the least number of movies that was released and what is the title of the movie in that year that has the highest number of rating score of 1?
Hint: least number of movies refers to MIN(movie_release_year); highest rating score refers to MAX(SUM(movie_id) where rating_score = '1')
A: Let’s think step by step. In the question , we are asked:
"Which year" so we need column = [movies.movie_release_year]
"number of movies" so we need column = [movies.movie_id]
"title of the movie" so we need column = [movies.movie_title]
"rating score" so we need column = [ratings.rating_score]
Hint also refers to the columns = [movies.movie_release_year, movies.movie_id, ratings.rating_score]
Based on the columns and tables, we need these Foreign_keys = [movies.movie_id = ratings.movie_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1]. So the Schema_links are:
Schema_links: [movies.movie_release_year, movies.movie_title, ratings.rating_score, movies.movie_id=ratings.movie_id, 1]

Schema of the database with sample rows:
#
CREATE TABLE lists (
        user_id INTEGER, 
        list_id INTEGER NOT NULL, 
        list_title TEXT, 
        list_movie_number INTEGER, 
        list_update_timestamp_utc TEXT, 
        list_creation_timestamp_utc TEXT, 
        list_followers INTEGER, 
        list_url TEXT, 
        list_comments INTEGER, 
        list_description TEXT, 
        list_cover_image_url TEXT, 
        list_first_image_url TEXT, 
        list_second_image_url TEXT, 
        list_third_image_url TEXT, 
        PRIMARY KEY (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id)
)

/*
3 rows from lists table:
user_id list_id list_title      list_movie_number       list_update_timestamp_utc       list_creation_timestamp_utc     list_followers  list_url        list_commentslist_description list_cover_image_url    list_first_image_url    list_second_image_url   list_third_image_url
88260493        1       Films that made your kid sister cry     5       2019-01-24 19:16:18     2009-11-11 00:02:21     5       http://mubi.com/lists/films-that-made-your-kid-sister-cry     3       <p>Don’t be such a baby!!</p>
<p><strong>bold</strong></p>    https://assets.mubicdn.net/images/film/3822/image-w1280.jpg?1445914994  https://assets.mubicdn.net/images/film/3822/image-w320.jpg?1445914994 https://assets.mubicdn.net/images/film/506/image-w320.jpg?1543838422    https://assets.mubicdn.net/images/film/485/image-w320.jpg?1575331204
45204418        2       Headscratchers  3       2018-12-03 15:12:20     2009-11-11 00:05:11     1       http://mubi.com/lists/headscratchers    2       <p>Films that need at least two viewings to really make sense.</p>
<p>Or at least… they did for <em>       https://assets.mubicdn.net/images/film/4343/image-w1280.jpg?1583331932  https://assets.mubicdn.net/images/film/4343/image-w320.jpg?1583331932 https://assets.mubicdn.net/images/film/159/image-w320.jpg?1548864573    https://assets.mubicdn.net/images/film/142/image-w320.jpg?1544094102
48905025        3       Sexy Time Movies        7       2019-05-30 03:00:07     2009-11-11 00:20:00     6       http://mubi.com/lists/sexy-time-movies  5       <p>Films that get you in the mood…for love. In development.</p>
<p>Remarks</p>
<p><strong>Enter the    https://assets.mubicdn.net/images/film/3491/image-w1280.jpg?1564112978  https://assets.mubicdn.net/images/film/3491/image-w320.jpg?1564112978https://assets.mubicdn.net/images/film/2377/image-w320.jpg?1564675204    https://assets.mubicdn.net/images/film/2874/image-w320.jpg?1546574412
*/

CREATE TABLE lists_users (
        user_id INTEGER NOT NULL, 
        list_id INTEGER NOT NULL, 
        list_update_date_utc TEXT, 
        list_creation_date_utc TEXT, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_avatar_image_url TEXT, 
        user_cover_image_url TEXT, 
        user_eligible_for_trial TEXT, 
        user_has_payment_method TEXT, 
        PRIMARY KEY (user_id, list_id), 
        FOREIGN KEY(list_id) REFERENCES lists (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists (user_id)
)

/*
3 rows from lists_users table:
user_id list_id list_update_date_utc    list_creation_date_utc  user_trialist   user_subscriber user_avatar_image_url   user_cover_image_url    user_eligible_for_trial       user_has_payment_method
85981819        1969    2019-11-26      2009-12-18      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        3946    2020-05-01      2010-01-30      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        6683    2020-04-12      2010-03-31      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
*/

Table: lists
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_title: column description -> Name of the list
Column list_movie_number: column description -> Number of movies added to the list
Column list_update_timestamp_utc: column description -> Last update timestamp for the list
Column list_creation_timestamp_utc: column description -> Creation timestamp for the list
Column list_followers: column description -> Number of followers on the list
Column list_url: column description -> URL to the list page on Mubi
Column list_comments: column description -> Number of comments on the list
Column list_description: column description -> List description made by the user

Table: lists_users
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_update_date_utc: column description -> Last update date for the list, value description -> YYYY-MM-DD
Column list_creation_date_utc: column description -> Creation date for the list, value description -> YYYY-MM-DD
Column user_trialist: column description -> whether the user was a tralist when he created the list , value description -> 1 = the user was a trialist when he created the list 0 = the user was not a trialist when he created the list
Column user_subscriber: column description -> whether the user was a subscriber when he created the list , value description -> 1 = the user was a subscriber when he created the list 0 = the user was not a subscriber when he created the list
Column user_avatar_image_url: column description -> User profile image URL on Mubi
Column user_cover_image_url: column description -> User profile cover image URL on Mubi
Column user_eligible_for_trial: column description -> whether the user was eligible for trial when he created the list , value description -> 1 = the user was eligible for trial when he created the list 0 = the user was not eligible for trial when he created the list
Column user_has_payment_method : column description -> whether the user was a paying subscriber when he created the list , value description -> 1 = the user was a paying subscriber when he created the list 0 = the user was not a paying subscriber when he created the list
#
Q: Among the lists created by user 4208563, which one has the highest number of followers? Indicate how many followers it has and whether the user was a subscriber or not when he created the list.
Hint: User 4208563 refers to user_id;highest number of followers refers to MAX(list_followers); user_subscriber = 1 means that the user was a subscriber when he created the list; user_subscriber = 0 means the user was not a subscriber when he created the list (to replace)
A: Let’s think step by step. In the question , we are asked:
"user" so we need column = [lists_users.user_id]
"number of followers" so we need column = [lists.list_followers]
"user was a subscriber or not" so we need column = [lists_users.user_subscriber]
Hint also refers to the columns = [lists_users.user_id,lists.list_followers,lists_users.user_subscriber]
Based on the columns and tables, we need these Foreign_keys = [lists.user_id = lists_user.user_id,lists.list_id = lists_user.list_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1, 4208563]. So the Schema_links are:
Schema_links: [lists.list_followers,lists_users.user_subscriber,lists.user_id = lists_user.user_id,lists.list_id = lists_user.list_id, lists_users.user_id, 4208563, 1]

"""  # noqa: E501

HUMAN_SCHEMA_LINKING_TEMPLATE = """
For the given question, find the schema links between the question and the table.
Hint helps you to fine the correct schema_links.
###
Schema of the database with sample rows and column descriptions:
#
{schema}

{columns_descriptions}
#
Q: {question}
Hint: {hint}
A: Let's think step by step. In the question , we are asked:
"""

SYSTEM_CLASSIFICATION_TEMPLATE = """
For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN.
if need nested queries: predict NESTED
elif need JOIN and don't need nested queries: predict NON-NESTED
elif don't need JOIN and don't need nested queries: predict EASY
Note: Don't mistake the WHERE conditions with nested queries.
Note: Only predict NESTED if the question needs nested queries, if it can be solved with JOIN, predict NON-NESTED.
###
Few examples of this task are:
###
Schema of the database with sample rows and column descriptions:
#
CREATE TABLE lists (
        user_id INTEGER, 
        list_id INTEGER NOT NULL, 
        list_title TEXT, 
        list_movie_number INTEGER, 
        list_update_timestamp_utc TEXT, 
        list_creation_timestamp_utc TEXT, 
        list_followers INTEGER, 
        list_url TEXT, 
        list_comments INTEGER, 
        list_description TEXT, 
        list_cover_image_url TEXT, 
        list_first_image_url TEXT, 
        list_second_image_url TEXT, 
        list_third_image_url TEXT, 
        PRIMARY KEY (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id)
)

/*
3 rows from lists table:
user_id list_id list_title      list_movie_number       list_update_timestamp_utc       list_creation_timestamp_utc     list_followers  list_url        list_commentslist_description list_cover_image_url    list_first_image_url    list_second_image_url   list_third_image_url
88260493        1       Films that made your kid sister cry     5       2019-01-24 19:16:18     2009-11-11 00:02:21     5       http://mubi.com/lists/films-that-made-your-kid-sister-cry     3       <p>Don’t be such a baby!!</p>
<p><strong>bold</strong></p>    https://assets.mubicdn.net/images/film/3822/image-w1280.jpg?1445914994  https://assets.mubicdn.net/images/film/3822/image-w320.jpg?1445914994 https://assets.mubicdn.net/images/film/506/image-w320.jpg?1543838422    https://assets.mubicdn.net/images/film/485/image-w320.jpg?1575331204
45204418        2       Headscratchers  3       2018-12-03 15:12:20     2009-11-11 00:05:11     1       http://mubi.com/lists/headscratchers    2       <p>Films that need at least two viewings to really make sense.</p>
<p>Or at least… they did for <em>       https://assets.mubicdn.net/images/film/4343/image-w1280.jpg?1583331932  https://assets.mubicdn.net/images/film/4343/image-w320.jpg?1583331932 https://assets.mubicdn.net/images/film/159/image-w320.jpg?1548864573    https://assets.mubicdn.net/images/film/142/image-w320.jpg?1544094102
48905025        3       Sexy Time Movies        7       2019-05-30 03:00:07     2009-11-11 00:20:00     6       http://mubi.com/lists/sexy-time-movies  5       <p>Films that get you in the mood…for love. In development.</p>
<p>Remarks</p>
<p><strong>Enter the    https://assets.mubicdn.net/images/film/3491/image-w1280.jpg?1564112978  https://assets.mubicdn.net/images/film/3491/image-w320.jpg?1564112978https://assets.mubicdn.net/images/film/2377/image-w320.jpg?1564675204    https://assets.mubicdn.net/images/film/2874/image-w320.jpg?1546574412
*/


CREATE TABLE lists_users (
        user_id INTEGER NOT NULL, 
        list_id INTEGER NOT NULL, 
        list_update_date_utc TEXT, 
        list_creation_date_utc TEXT, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_avatar_image_url TEXT, 
        user_cover_image_url TEXT, 
        user_eligible_for_trial TEXT, 
        user_has_payment_method TEXT, 
        PRIMARY KEY (user_id, list_id), 
        FOREIGN KEY(list_id) REFERENCES lists (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists (user_id)
)

/*
3 rows from lists_users table:
user_id list_id list_update_date_utc    list_creation_date_utc  user_trialist   user_subscriber user_avatar_image_url   user_cover_image_url    user_eligible_for_trial       user_has_payment_method
85981819        1969    2019-11-26      2009-12-18      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        3946    2020-05-01      2010-01-30      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        6683    2020-04-12      2010-03-31      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
*/

CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)

/*
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/

CREATE TABLE ratings (
        movie_id INTEGER, 
        rating_id INTEGER, 
        rating_url TEXT, 
        rating_score INTEGER, 
        rating_timestamp_utc TEXT, 
        critic TEXT, 
        critic_likes INTEGER, 
        critic_comments INTEGER, 
        user_id INTEGER, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_eligible_for_trial INTEGER, 
        user_has_payment_method INTEGER, 
        FOREIGN KEY(movie_id) REFERENCES movies (movie_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id), 
        FOREIGN KEY(rating_id) REFERENCES ratings (rating_id), 
        FOREIGN KEY(user_id) REFERENCES ratings_users (user_id)
)

/*
3 rows from ratings table:
movie_id        rating_id       rating_url      rating_score    rating_timestamp_utc    critic  critic_likes    critic_comments user_id user_trialist   user_subscriber       user_eligible_for_trial user_has_payment_method
1066    15610495        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/15610495 3       2017-06-10 12:38:33     None    0       0       41579158     00       1       0
1066    10704606        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10704606 2       2014-08-15 23:42:31     None    0       0       85981819     11       0       1
1066    10177114        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10177114 2       2014-01-30 13:21:57     None    0       0       4208563 0    01       1
*/

Table: lists_users
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_update_date_utc: column description -> Last update date for the list, value description -> YYYY-MM-DD
Column list_creation_date_utc: column description -> Creation date for the list, value description -> YYYY-MM-DD
Column user_trialist: column description -> whether the user was a tralist when he created the list , value description -> 1 = the user was a trialist when he created the list 0 = the user was not a trialist when he created the list
Column user_subscriber: column description -> whether the user was a subscriber when he created the list , value description -> 1 = the user was a subscriber when he created the list 0 = the user was not a subscriber when he created the list
Column user_avatar_image_url: column description -> User profile image URL on Mubi
Column user_cover_image_url: column description -> User profile cover image URL on Mubi
Column user_eligible_for_trial: column description -> whether the user was eligible for trial when he created the list , value description -> 1 = the user was eligible for trial when he created the list 0 = the user was not eligible for trial when he created the list
Column user_has_payment_method : column description -> whether the user was a paying subscriber when he created the list , value description -> 1 = the user was a paying subscriber when he created the list 0 = the user was not a paying subscriber when he created the list 

Table: lists
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_title: column description -> Name of the list
Column list_movie_number: column description -> Number of movies added to the list
Column list_update_timestamp_utc: column description -> Last update timestamp for the list
Column list_creation_timestamp_utc: column description -> Creation timestamp for the list
Column list_followers: column description -> Number of followers on the list
Column list_url: column description -> URL to the list page on Mubi
Column list_comments: column description -> Number of comments on the list
Column list_description: column description -> List description made by the user

Table: ratings
Column movie_id: column description -> Movie ID related to the rating
Column rating_id: column description -> Rating ID on Mubi
Column rating_url: column description -> URL to the rating on Mubi
Column rating_score: column description -> Rating score ranging from 1 (lowest) to 5 (highest), value description -> commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie
Column rating_timestamp_utc : column description -> Timestamp for the movie rating made by the user on Mubi
Column critic: column description -> Critic made by the user rating the movie. , value description -> If value = "None", the user did not write a critic when rating the movie.
Column critic_likes: column description -> Number of likes related to the critic made by the user rating the movie
Column critic_comments: column description -> Number of comments related to the critic made by the user rating the movie
Column user_id: column description -> ID related to the user rating the movie
Column user_trialist : column description -> whether user was a tralist when he rated the movie, value description -> 1 = the user was a trialist when he rated the movie 0 = the user was not a trialist when he rated the movie

Table: movies
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi
#

Q: What is the list ID that was first created by user 85981819?
Hint: first created list refers to oldest list_creation_date_utc;
schema_links: [lists_users.list_id, lists_users.user_id, lists_user.list_creation_date_utc, 85981819]
A: Let’s think step by step. The SQL query for the given question needs these tables = [lists_users], so we don't need JOIN.
Plus, it doesn't require nested queries, and we need the answer to the sub-questions = [""].
So, we don't need JOIN and don't need nested queries, then the SQL query can be classified as "EASY".
Label: "EASY"

Q: How many more movie lists were created by the user who created the movie list \"250 Favourite Films\"?
Hint: 250 Favourite Films refers to list_title;
schema_links: [lists_users.list_id,lists_users.user_id,lists.user_id,lists.list_title,250 Favourite Films]
A: Let’s think step by step. The SQL query for the given question needs these tables = [lists,lists_users], so we need JOIN.
Plus, it requires nested queries, and we need the answer to the sub-questions = [who created the movie list \"250 Favourite Films\"?].
So, we need JOIN and need nested queries, then the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: What is the percentage of the ratings were rated by user who was a subcriber?
Hint: user is a subscriber refers to user_subscriber = 1; percentage of ratings = DIVIDE(SUM(user_subscriber = 1), SUM(rating_score)) as percent;
schema_links: [ratings.user_subscriber,1]
A: Let’s think step by step. The SQL query for the given question needs these tables = [ratings], so we don't need JOIN.
Plus, it doesn't require nested queries, and we need the answer to the sub-questions = [""].
So, we don't need JOIN and don't need nested queries, then the SQL query can be classified as "EASY".
Label: "EASY"

Q: Was the user who created the \"World War 2 and Kids\" list eligible for trial when he created the list? Indicate how many followers does the said list has.
Hint: user was eligible for trial when he created the list refers to user_eligible_for_trial = 1; number of followers a list have refers to list_followers;
schema_links: [lists_users.user_eligible_for_trial, lists.list_followers, lists.list_title, lists.user_id = lists_user.user_id,lists.list_id = lists_user.list_id, World War 2 and Kids]
A: Let’s think step by step. The SQL query for the given question needs these tables = [lists, lists_users], so we need JOIN.
Plus, it doesn't need nested queries, and we need the answer to the sub-questions = [""].
So, we need JOIN and don't need nested queries, then the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: Which year was the third movie directed by Quentin Tarantino released? Indicate the user ids of the user who gave it a rating score of 4.
Hint: third movie refers to third movie that has oldest movie_release_year;
schema_links: [movies.movie_release_year,ratings.user_id,ratings.rating_score,movies.movie_id = ratings.movie_id, movies.director_name, Quentin Tarantino, 4]
A: Let’s think step by step. The SQL query for the given question needs these tables = [ratings,movies], so we need JOIN.
Plus, it requires nested queries, and we need the answer to the sub-questions = [Which movie is the third movie directed by Quentin Tarantino?].
So, we need JOIN and need nested queries, then the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: What is the average number of followers of the lists created by the user who rated the movie \"Pavee Lackeen: The Traveller Girl\" on 3/27/2011 at 2:06:34 AM?
Hint: average number of followers refers to AVG(list_followers); movie \"Pavee Lackeen: The Traveller Girl\" refers to movie_title = 'Pavee Lackeen: The Traveller Girl'; on 3/27/2011 at 2:06:34 AM refers to rating_timestamp_utc = '2011-03-27 02:06:34'
schema_links: [ratings.rating_timestamp_utc,lists_users.list_id,movies.movie_title,lists.list_followers,ratings.user_id = list_user.user_id,ratings.movie_id = movies.movie_id,lists_users.list_id = lists.list_id,Pavee Lackeen: The Traveller Girl,2011-03-27 02:06:34]
A: Let’s think step by step. The SQL query for the given question needs these tables = [lists, lists_users,ratings,movies], so we need JOIN.
Plus, it doesn't need nested queries, and we need the answer to the sub-questions = [""].
So, we need JOIN and don't need nested queries, then the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

"""  # noqa: E501
HUMAN_CLASSIFICATION_TEMPLATE = """
For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN.
if need nested queries: predict NESTED
elif need JOIN and don't need nested queries: predict NON-NESTED
elif don't need JOIN and don't need nested queries: predict EASY
###
Schema of the database with sample rows and column descriptions:
#
{schema}

{columns_descriptions}
#
Q: {question}
Hint: {hint}
Schema links: {schema_links}
A: Let’s think step by step."""  # noqa: E501

SYSTEM_EASY_CLASS_TEMPLATE = """
Use the schema links to generate the correct sqlite SQL query for the given question.
Hint helps you to write the correct sqlite SQL query.
###
Few examples of this task are:
###
Schema of the database with sample rows and column descriptions:
#
CREATE TABLE lists (
        user_id INTEGER, 
        list_id INTEGER NOT NULL, 
        list_title TEXT, 
        list_movie_number INTEGER, 
        list_update_timestamp_utc TEXT, 
        list_creation_timestamp_utc TEXT, 
        list_followers INTEGER, 
        list_url TEXT, 
        list_comments INTEGER, 
        list_description TEXT, 
        list_cover_image_url TEXT, 
        list_first_image_url TEXT, 
        list_second_image_url TEXT, 
        list_third_image_url TEXT, 
        PRIMARY KEY (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id)
)

/*
3 rows from lists table:
user_id list_id list_title      list_movie_number       list_update_timestamp_utc       list_creation_timestamp_utc     list_followers  list_url        list_commentslist_description list_cover_image_url    list_first_image_url    list_second_image_url   list_third_image_url
88260493        1       Films that made your kid sister cry     5       2019-01-24 19:16:18     2009-11-11 00:02:21     5       http://mubi.com/lists/films-that-made-your-kid-sister-cry     3       <p>Don’t be such a baby!!</p>
<p><strong>bold</strong></p>    https://assets.mubicdn.net/images/film/3822/image-w1280.jpg?1445914994  https://assets.mubicdn.net/images/film/3822/image-w320.jpg?1445914994 https://assets.mubicdn.net/images/film/506/image-w320.jpg?1543838422    https://assets.mubicdn.net/images/film/485/image-w320.jpg?1575331204
45204418        2       Headscratchers  3       2018-12-03 15:12:20     2009-11-11 00:05:11     1       http://mubi.com/lists/headscratchers    2       <p>Films that need at least two viewings to really make sense.</p>
<p>Or at least… they did for <em>       https://assets.mubicdn.net/images/film/4343/image-w1280.jpg?1583331932  https://assets.mubicdn.net/images/film/4343/image-w320.jpg?1583331932 https://assets.mubicdn.net/images/film/159/image-w320.jpg?1548864573    https://assets.mubicdn.net/images/film/142/image-w320.jpg?1544094102
48905025        3       Sexy Time Movies        7       2019-05-30 03:00:07     2009-11-11 00:20:00     6       http://mubi.com/lists/sexy-time-movies  5       <p>Films that get you in the mood…for love. In development.</p>
<p>Remarks</p>
<p><strong>Enter the    https://assets.mubicdn.net/images/film/3491/image-w1280.jpg?1564112978  https://assets.mubicdn.net/images/film/3491/image-w320.jpg?1564112978https://assets.mubicdn.net/images/film/2377/image-w320.jpg?1564675204    https://assets.mubicdn.net/images/film/2874/image-w320.jpg?1546574412
*/

CREATE TABLE lists_users (
        user_id INTEGER NOT NULL, 
        list_id INTEGER NOT NULL, 
        list_update_date_utc TEXT, 
        list_creation_date_utc TEXT, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_avatar_image_url TEXT, 
        user_cover_image_url TEXT, 
        user_eligible_for_trial TEXT, 
        user_has_payment_method TEXT, 
        PRIMARY KEY (user_id, list_id), 
        FOREIGN KEY(list_id) REFERENCES lists (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists (user_id)
)

/*
3 rows from lists_users table:
user_id list_id list_update_date_utc    list_creation_date_utc  user_trialist   user_subscriber user_avatar_image_url   user_cover_image_url    user_eligible_for_trial       user_has_payment_method
85981819        1969    2019-11-26      2009-12-18      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        3946    2020-05-01      2010-01-30      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        6683    2020-04-12      2010-03-31      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
*/

CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)

/*
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/

CREATE TABLE ratings (
        movie_id INTEGER, 
        rating_id INTEGER, 
        rating_url TEXT, 
        rating_score INTEGER, 
        rating_timestamp_utc TEXT, 
        critic TEXT, 
        critic_likes INTEGER, 
        critic_comments INTEGER, 
        user_id INTEGER, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_eligible_for_trial INTEGER, 
        user_has_payment_method INTEGER, 
        FOREIGN KEY(movie_id) REFERENCES movies (movie_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id), 
        FOREIGN KEY(rating_id) REFERENCES ratings (rating_id), 
        FOREIGN KEY(user_id) REFERENCES ratings_users (user_id)
)

/*
3 rows from ratings table:
movie_id        rating_id       rating_url      rating_score    rating_timestamp_utc    critic  critic_likes    critic_comments user_id user_trialist   user_subscriber       user_eligible_for_trial user_has_payment_method
1066    15610495        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/15610495 3       2017-06-10 12:38:33     None    0       0       41579158     00       1       0
1066    10704606        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10704606 2       2014-08-15 23:42:31     None    0       0       85981819     11       0       1
1066    10177114        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10177114 2       2014-01-30 13:21:57     None    0       0       4208563 0    01       1
*/

Table: lists_users
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_update_date_utc: column description -> Last update date for the list, value description -> YYYY-MM-DD
Column list_creation_date_utc: column description -> Creation date for the list, value description -> YYYY-MM-DD
Column user_trialist: column description -> whether the user was a tralist when he created the list , value description -> 1 = the user was a trialist when he created the list 0 = the user was not a trialist when he created the list
Column user_subscriber: column description -> whether the user was a subscriber when he created the list , value description -> 1 = the user was a subscriber when he created the list 0 = the user was not a subscriber when he created the list
Column user_avatar_image_url: column description -> User profile image URL on Mubi
Column user_cover_image_url: column description -> User profile cover image URL on Mubi
Column user_eligible_for_trial: column description -> whether the user was eligible for trial when he created the list , value description -> 1 = the user was eligible for trial when he created the list 0 = the user was not eligible for trial when he created the list
Column user_has_payment_method : column description -> whether the user was a paying subscriber when he created the list , value description -> 1 = the user was a paying subscriber when he created the list 0 = the user was not a paying subscriber when he created the list 

Table: lists
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_title: column description -> Name of the list
Column list_movie_number: column description -> Number of movies added to the list
Column list_update_timestamp_utc: column description -> Last update timestamp for the list
Column list_creation_timestamp_utc: column description -> Creation timestamp for the list
Column list_followers: column description -> Number of followers on the list
Column list_url: column description -> URL to the list page on Mubi
Column list_comments: column description -> Number of comments on the list
Column list_description: column description -> List description made by the user

Table: ratings
Column movie_id: column description -> Movie ID related to the rating
Column rating_id: column description -> Rating ID on Mubi
Column rating_url: column description -> URL to the rating on Mubi
Column rating_score: column description -> Rating score ranging from 1 (lowest) to 5 (highest), value description -> commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie
Column rating_timestamp_utc : column description -> Timestamp for the movie rating made by the user on Mubi
Column critic: column description -> Critic made by the user rating the movie. , value description -> If value = "None", the user did not write a critic when rating the movie.
Column critic_likes: column description -> Number of likes related to the critic made by the user rating the movie
Column critic_comments: column description -> Number of comments related to the critic made by the user rating the movie
Column user_id: column description -> ID related to the user rating the movie
Column user_trialist : column description -> whether user was a tralist when he rated the movie, value description -> 1 = the user was a trialist when he rated the movie 0 = the user was not a trialist when he rated the movie

Table: movies
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi
#
Q: What is the name of the longest movie title? When was it released?
Hint: longest movie title refers to MAX(LENGTH(movie_title)); when it was released refers to movie_release_year;
Schema_links: [movies.movie_title,movies.movie_release_year, movies.movie_popularity]
SQL: SELECT movie_title, movie_release_year FROM movies ORDER BY LENGTH(movie_popularity) DESC LIMIT 1

Q: What is the percentage of the ratings were rated by user who was a subcriber?
Hint: user is a subscriber refers to user_subscriber = 1; percentage of ratings = DIVIDE(SUM(user_subscriber = 1), SUM(rating_score)) as percent;
Schema_links: [ratings.user_subscriber,1]
SQL: SELECT CAST(SUM(CASE WHEN user_subscriber = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings

Q: When was the first movie released and who directed it?
Hint: first movie refers to oldest movie_release_year;
Schema_links: [movies.movie_release_year, movies.director_name]
SQL: SELECT movie_release_year, director_name FROM movies WHERE movie_release_year IS NOT NULL ORDER BY movie_release_year ASC LIMIT 1

Q: How many movie lists were still updated 10 years after it was created?
Hint: updated 10 years after it was created refers to list_update_timestamp_utc > (list_creation_timestamp_utc+10);
Schema_links: [lists.list_update_timestamp_utc, lists.list_creation_timestamp_utc, 10]
SQL: SELECT COUNT(*) FROM lists WHERE SUBSTR(list_update_timestamp_utc, 1, 4) - SUBSTR(list_creation_timestamp_utc, 1, 4) > 10

Q: For the list with more than 200 followers, state the title and how long the list has been created?
Hint: more than 200 followers refers to list_followers >200; how long the list has been created refers to SUBTRACT(CURRENT_TIMESTAMP,list_creation_timestamp_utc)
Schema_links: [lists.list_title, lists.list_update_timestamp_utc,lists.list_followers, 200]
SQL: SELECT list_title , 365 * (strftime('%Y', 'now') - strftime('%Y', list_creation_timestamp_utc)) + 30 * (strftime('%m', 'now') - strftime('%m', list_creation_timestamp_utc)) + strftime('%d', 'now') - strftime('%d', list_creation_timestamp_utc) FROM lists WHERE list_followers > 200

Q: What is the percentage of list created by user who was a subscriber when he created the list?
Hint: was a subscriber refers to user_subscriber = 1; percentage refers to DIVIDE(COUNT(user_subscriber = 1),COUNT(list_id))
Schema_links: [lists_users.user_subscriber,1]
SQL: SELECT CAST(SUM(CASE WHEN user_subscriber = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(list_id) FROM lists_users

"""  # noqa: E501
HUMAN_EASY_CLASS_TEMPLATE = """
Use the schema links to generate the correct sqlite SQL query for the given question.
Hint helps you to write the correct sqlite SQL query.
###
Schema of the database with sample rows and column descriptions:
#
{schema}

{columns_descriptions}
#
Q: {question}
Hint: {hint}
Schema_links: {schema_links}
SQL: """


SYSTEM_NON_NESTED_CLASS_TEMPLATE = """
Use the the schema links and intermediate reasoning steps to generate the correct sqlite SQL query for the given question.
Hint helps you to write the correct sqlite SQL query.
###
Few examples of this task are:
###
Schema of the database with sample rows and column descriptions:
#
CREATE TABLE ratings (
        movie_id INTEGER, 
        rating_id INTEGER, 
        rating_url TEXT, 
        rating_score INTEGER, 
        rating_timestamp_utc TEXT, 
        critic TEXT, 
        critic_likes INTEGER, 
        critic_comments INTEGER, 
        user_id INTEGER, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_eligible_for_trial INTEGER, 
        user_has_payment_method INTEGER, 
        FOREIGN KEY(movie_id) REFERENCES movies (movie_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id), 
        FOREIGN KEY(rating_id) REFERENCES ratings (rating_id), 
        FOREIGN KEY(user_id) REFERENCES ratings_users (user_id)
)

/*
3 rows from ratings table:
movie_id        rating_id       rating_url      rating_score    rating_timestamp_utc    critic  critic_likes    critic_comments user_id user_trialist   user_subscriber       user_eligible_for_trial user_has_payment_method
1066    15610495        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/15610495 3       2017-06-10 12:38:33     None    0       0       41579158     00       1       0
1066    10704606        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10704606 2       2014-08-15 23:42:31     None    0       0       85981819     11       0       1
1066    10177114        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10177114 2       2014-01-30 13:21:57     None    0       0       4208563 0    01       1
*/

CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)

/*
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/

Table: ratings
Column movie_id: column description -> Movie ID related to the rating
Column rating_id: column description -> Rating ID on Mubi
Column rating_url: column description -> URL to the rating on Mubi
Column rating_score: column description -> Rating score ranging from 1 (lowest) to 5 (highest), value description -> commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie
Column rating_timestamp_utc : column description -> Timestamp for the movie rating made by the user on Mubi
Column critic: column description -> Critic made by the user rating the movie. , value description -> If value = "None", the user did not write a critic when rating the movie.
Column critic_likes: column description -> Number of likes related to the critic made by the user rating the movie
Column critic_comments: column description -> Number of comments related to the critic made by the user rating the movie
Column user_id: column description -> ID related to the user rating the movie
Column user_trialist : column description -> whether user was a tralist when he rated the movie, value description -> 1 = the user was a trialist when he rated the movie 0 = the user was not a trialist when he rated the movie

Table: movies
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi
#
Q: What is the average rating for movie titled 'When Will I Be Loved'?
Hint: average rating = DIVIDE((SUM(rating_score where movie_title = 'When Will I Be Loved')), COUNT(rating_score));
Schema_links: [ratings.rating_score, movies.movie_title, movies.movie_id = ratings.movie_id, When Will I Be Loved]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [ratings,movies].
First of all, for joining these tables we have to use the common column = [ratings.movie_id = movies.movie_id].
Now, we have to filter the rows where movie_title = 'When Will I Be Loved'.
Then, we have to find the average of the rating_score.
So the sqlite SQL query will be:
SQL: SELECT AVG(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'When Will I Be Loved'

Q: For movie titled 'Welcome to the Dollhouse', how many percentage of the ratings were rated with highest score.
Hint: rated with highest score refers to rating_score = 5; percentage = MULTIPLY(DIVIDE(SUM(rating_score = 5), COUNT(rating_score)), 100)
Schema_links: [ratings.rating_score, movies.movie_title, movies.movie_id = ratings.movie_id, Welcome to the Dollhouse]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [ratings,movies].
First of all, for joining these tables we have to use the common column = [ratings.movie_id = movies.movie_id].
Now, we have to filter the rows where movie_title = 'Welcome to the Dollhouse'.
Then, we have to find the percentage of the ratings were rated with highest score which is 5.
So the sqlite SQL query will be:
SQL: SELECT CAST(SUM(CASE WHEN T2.rating_score = 5 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'Welcome to the Dollhouse'

Q: For all ratings which are rated in year 2020, name the movies which has the rating scored 4 and above.
Hint: ratings in year 2020 refers to rating_timestamp_utc like '%2020%'; rating_score > = 4;
Schema_links: [ratings.rating_timestamp_utc, movies.movie_title, movies.movie_id = ratings.movie_id, 2020, 4]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [ratings,movies].
First of all, for joining these tables we have to use the common column = [ratings.movie_id = movies.movie_id].
Now, we have to filter the rows where rating_timestamp_utc like '%2020%' and rating_score > = 4.
Then, we have to find the movie_title.
So the sqlite SQL query will be:
SQL: SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE CAST(SUBSTR(T1.rating_timestamp_utc, 1, 4) AS INTEGER) = 2020 AND CAST(SUBSTR(T1.rating_timestamp_utc, 6, 2) AS INTEGER) > 4

Q: What is the average score of the movie \"The Fall of Berlin\" in 2019?
Hint: The Fall of Berlin' is movie_title; in 2019 refers to rating_timestamp_utc = 2019; Average score refers to Avg(rating_score);
Schema_links: [ratings.rating_score, ratings.rating_id, movies.movie_title, T1.rating_timestamp_utc, movies.movie_id = ratings.movie_id, The Fall of Berlin, 2019]
A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [ratings,movies].
First of all, for joining these tables we have to use the common column = [ratings.movie_id = movies.movie_id].
Now, we have to filter the rows where movie_title = 'The Fall of Berlin' and rating_timestamp_utc = 2019.
Then, we have to find the average of the rating_score which can be computed by dividing the sum of rating_score by the count of rating_id.
So the sqlite SQL query will be:
SQL: SELECT SUM(T1.rating_score) / COUNT(T1.rating_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_timestamp_utc LIKE '2019%' AND T2.movie_title LIKE 'The Fall of Berlin'

"""  # noqa: E501
HUMAN_NON_NESTED_CLASS_TEMPLATE = """
Use the the schema links and intermediate reasoning steps to generate the correct sqlite SQL query for the given question.
Hint helps you to write the correct sqlite SQL query.
###
Schema of the database with sample rows and column descriptions:
#
{schema}

{columns_descriptions}
#
Q: {question}
Hint: {hint}
Schema_links: {schema_links}
A: Let’s think step by step. """  # noqa: E501

SYSTEM_NESTED_CLASS_TEMPLATE = """
Use the the schema links and intermediate reasoning steps to generate the correct sqlite SQL query for the given question.
Hint helps you to write the correct sqlite SQL query.
###
Few examples of this task are:
###
Schema of the database with sample rows and column descriptions:
#
CREATE TABLE lists (
        user_id INTEGER, 
        list_id INTEGER NOT NULL, 
        list_title TEXT, 
        list_movie_number INTEGER, 
        list_update_timestamp_utc TEXT, 
        list_creation_timestamp_utc TEXT, 
        list_followers INTEGER, 
        list_url TEXT, 
        list_comments INTEGER, 
        list_description TEXT, 
        list_cover_image_url TEXT, 
        list_first_image_url TEXT, 
        list_second_image_url TEXT, 
        list_third_image_url TEXT, 
        PRIMARY KEY (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id)
)

/*
3 rows from lists table:
user_id list_id list_title      list_movie_number       list_update_timestamp_utc       list_creation_timestamp_utc     list_followers  list_url        list_commentslist_description list_cover_image_url    list_first_image_url    list_second_image_url   list_third_image_url
88260493        1       Films that made your kid sister cry     5       2019-01-24 19:16:18     2009-11-11 00:02:21     5       http://mubi.com/lists/films-that-made-your-kid-sister-cry     3       <p>Don’t be such a baby!!</p>
<p><strong>bold</strong></p>    https://assets.mubicdn.net/images/film/3822/image-w1280.jpg?1445914994  https://assets.mubicdn.net/images/film/3822/image-w320.jpg?1445914994 https://assets.mubicdn.net/images/film/506/image-w320.jpg?1543838422    https://assets.mubicdn.net/images/film/485/image-w320.jpg?1575331204
45204418        2       Headscratchers  3       2018-12-03 15:12:20     2009-11-11 00:05:11     1       http://mubi.com/lists/headscratchers    2       <p>Films that need at least two viewings to really make sense.</p>
<p>Or at least… they did for <em>       https://assets.mubicdn.net/images/film/4343/image-w1280.jpg?1583331932  https://assets.mubicdn.net/images/film/4343/image-w320.jpg?1583331932 https://assets.mubicdn.net/images/film/159/image-w320.jpg?1548864573    https://assets.mubicdn.net/images/film/142/image-w320.jpg?1544094102
48905025        3       Sexy Time Movies        7       2019-05-30 03:00:07     2009-11-11 00:20:00     6       http://mubi.com/lists/sexy-time-movies  5       <p>Films that get you in the mood…for love. In development.</p>
<p>Remarks</p>
<p><strong>Enter the    https://assets.mubicdn.net/images/film/3491/image-w1280.jpg?1564112978  https://assets.mubicdn.net/images/film/3491/image-w320.jpg?1564112978https://assets.mubicdn.net/images/film/2377/image-w320.jpg?1564675204    https://assets.mubicdn.net/images/film/2874/image-w320.jpg?1546574412
*/


CREATE TABLE lists_users (
        user_id INTEGER NOT NULL, 
        list_id INTEGER NOT NULL, 
        list_update_date_utc TEXT, 
        list_creation_date_utc TEXT, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_avatar_image_url TEXT, 
        user_cover_image_url TEXT, 
        user_eligible_for_trial TEXT, 
        user_has_payment_method TEXT, 
        PRIMARY KEY (user_id, list_id), 
        FOREIGN KEY(list_id) REFERENCES lists (list_id), 
        FOREIGN KEY(user_id) REFERENCES lists (user_id)
)

/*
3 rows from lists_users table:
user_id list_id list_update_date_utc    list_creation_date_utc  user_trialist   user_subscriber user_avatar_image_url   user_cover_image_url    user_eligible_for_trial       user_has_payment_method
85981819        1969    2019-11-26      2009-12-18      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        3946    2020-05-01      2010-01-30      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
85981819        6683    2020-04-12      2010-03-31      1       1       https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214      None    0    1
*/

CREATE TABLE ratings (
        movie_id INTEGER, 
        rating_id INTEGER, 
        rating_url TEXT, 
        rating_score INTEGER, 
        rating_timestamp_utc TEXT, 
        critic TEXT, 
        critic_likes INTEGER, 
        critic_comments INTEGER, 
        user_id INTEGER, 
        user_trialist INTEGER, 
        user_subscriber INTEGER, 
        user_eligible_for_trial INTEGER, 
        user_has_payment_method INTEGER, 
        FOREIGN KEY(movie_id) REFERENCES movies (movie_id), 
        FOREIGN KEY(user_id) REFERENCES lists_users (user_id), 
        FOREIGN KEY(rating_id) REFERENCES ratings (rating_id), 
        FOREIGN KEY(user_id) REFERENCES ratings_users (user_id)
)

/*
3 rows from ratings table:
movie_id        rating_id       rating_url      rating_score    rating_timestamp_utc    critic  critic_likes    critic_comments user_id user_trialist   user_subscriber       user_eligible_for_trial user_has_payment_method
1066    15610495        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/15610495 3       2017-06-10 12:38:33     None    0       0       41579158     00       1       0
1066    10704606        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10704606 2       2014-08-15 23:42:31     None    0       0       85981819     11       0       1
1066    10177114        http://mubi.com/films/pavee-lackeen-the-traveller-girl/ratings/10177114 2       2014-01-30 13:21:57     None    0       0       4208563 0    01       1
*/

CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)

/*
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/

Table: lists_users
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_update_date_utc: column description -> Last update date for the list, value description -> YYYY-MM-DD
Column list_creation_date_utc: column description -> Creation date for the list, value description -> YYYY-MM-DD
Column user_trialist: column description -> whether the user was a tralist when he created the list , value description -> 1 = the user was a trialist when he created the list 0 = the user was not a trialist when he created the list
Column user_subscriber: column description -> whether the user was a subscriber when he created the list , value description -> 1 = the user was a subscriber when he created the list 0 = the user was not a subscriber when he created the list
Column user_avatar_image_url: column description -> User profile image URL on Mubi
Column user_cover_image_url: column description -> User profile cover image URL on Mubi
Column user_eligible_for_trial: column description -> whether the user was eligible for trial when he created the list , value description -> 1 = the user was eligible for trial when he created the list 0 = the user was not eligible for trial when he created the list
Column user_has_payment_method : column description -> whether the user was a paying subscriber when he created the list , value description -> 1 = the user was a paying subscriber when he created the list 0 = the user was not a paying subscriber when he created the list 

Table: lists
Column user_id: column description -> ID related to the user who created the list.
Column list_id: column description -> ID of the list on Mubi
Column list_title: column description -> Name of the list
Column list_movie_number: column description -> Number of movies added to the list
Column list_update_timestamp_utc: column description -> Last update timestamp for the list
Column list_creation_timestamp_utc: column description -> Creation timestamp for the list
Column list_followers: column description -> Number of followers on the list
Column list_url: column description -> URL to the list page on Mubi
Column list_comments: column description -> Number of comments on the list
Column list_description: column description -> List description made by the user

Table: ratings
Column movie_id: column description -> Movie ID related to the rating
Column rating_id: column description -> Rating ID on Mubi
Column rating_url: column description -> URL to the rating on Mubi
Column rating_score: column description -> Rating score ranging from 1 (lowest) to 5 (highest), value description -> commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie
Column rating_timestamp_utc : column description -> Timestamp for the movie rating made by the user on Mubi
Column critic: column description -> Critic made by the user rating the movie. , value description -> If value = "None", the user did not write a critic when rating the movie.
Column critic_likes: column description -> Number of likes related to the critic made by the user rating the movie
Column critic_comments: column description -> Number of comments related to the critic made by the user rating the movie
Column user_id: column description -> ID related to the user rating the movie
Column user_trialist : column description -> whether user was a tralist when he rated the movie, value description -> 1 = the user was a trialist when he rated the movie 0 = the user was not a trialist when he rated the movie

Table: movies
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi
#
Q: How many more movie lists were created by the user who created the movie list \"250 Favourite Films\"?
Hint: 250 Favourite Films refers to list_title;
Schema_links: [lists_users.list_id, lists_users.user_id = lists.user_id, lists.list_title, 250 Favourite Films]
A: Let's think step by step. the given question can be solved by knowing the answer to the following sub-questions = [which user has created the movie list \"250 Favourite Films\".]
The sqlite SQL query for the sub-question "which user has created the movie list \"250 Favourite Films\"" is SELECT user_id FROM lists WHERE list_title = '250 Favourite Films'
The above query will return the user_id of the user who has created the movie list \"250 Favourite Films\".
Now, we have to find the number of movie lists created by the user who has created the movie list \"250 Favourite Films\".
So, the final sqlite SQL query answer to the question the given question is =
SQL: SELECT COUNT(list_id) FROM lists_users WHERE user_id = ( SELECT user_id FROM lists WHERE list_title = '250 Favourite Films' )

Q: For the user who post the list that contained the most number of the movies, is he/she a paying subscriber when creating that list?
Hint: the list that contained the most number of the movies refers to MAX(list_movie_number); user_has_payment_method = 1 means the user was a paying subscriber when he created the list ; \nuser_has_payment_method = 0 means the user was not a paying subscriber when he created the list
Schema_links: [lists_users.user_has_payment_method, lists_users.list_id = lists.list_id, lists.list_movie_number, lists.list_movie_number]
A: Let's think step by step. the given question can be solved by knowing the answer to the following sub-questions = [which list has the most number of movies.]
The sqlite SQL query for the sub-question "which list has the most number of movies" is SELECT MAX(list_movie_number) FROM lists
The above query will return the list_movie_number of the list which has the most number of movies.
Now, we have to find the user_has_payment_method of the user who has created the list which has the most number of movies.
To do so, we have to JOIN lists_users and lists table on list_id.
So, the final sqlite SQL query answer to the question the given question is =
SQL: SELECT T1.user_has_payment_method FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_movie_number = ( SELECT MAX(list_movie_number) FROM lists )

Q: Which year was the third movie directed by Quentin Tarantino released? Indicate the user ids of the user who gave it a rating score of 4.
Hint: third movie refers to third movie that has oldest movie_release_year;
Schema_links: [movies.movie_release_year,ratings.user_id,ratings.rating_score,movies.movie_id = ratings.movie_id, movies.director_name, Quentin Tarantino, 4]
A: Let's think step by step. the given question can be solved by knowing the answer to the following sub-questions = [What is the third movie directed by Quentin Tarantino.]
The sqlite SQL query for the sub-question "what is third movie directed by Quentin Tarantino" is SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino' ORDER BY movie_release_year ASC LIMIT 2, 1 
The above query will return the movie_id of the third movie directed by Quentin Tarantino.
Now, we have to find the year in which the third movie directed by Quentin Tarantino was released.
For that, we have to join the tables = [movies,ratings].
First of all, for joining these tables we have to use the common column = [movies.movie_id = ratings.movie_id].
Then, we have to filter the rows where movie_id = ( SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino' ORDER BY movie_release_year ASC LIMIT 2, 1 ).
Then, we have to find the movie_release_year.
So, the final sqlite SQL query answer to the question the given question is =
SQL: SELECT T2.movie_release_year, T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_id = ( SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino' ORDER BY movie_release_year ASC LIMIT 2, 1 ) AND T1.rating_score = 4

"""  # noqa: E501
HUMAN_NESTED_CLASS_TEMPLATE = """
Use the the schema links and intermediate reasoning steps to generate the correct sqlite SQL query for the given question.
Hint helps you to write the correct sqlite SQL query.
###
Schema of the database with sample rows and column descriptions:
#
{schema}

{columns_descriptions}
#
Q: {question}
Hint: {hint}
Schema_links: {schema_links}
A: Let's think step by step. the given question can be solved by knowing the answer to the following sub-questions = {sub_questions}
"""  # noqa: E501

SYSTEM_SELF_CORRECTION_PROMPT = """
For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQLite SQL QUERY as is.
Hint helps you to write the correct sqlite SQL query.
Use the following instructions for fixing the sqlite SQL query:
1) Avoid redundant columns in SELECT clause, all of the columns should be mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by checking the Foreign keys.
3) Pay attention to the columns that are used for the WHERE statement.
4) Pay attention to the columns that are used for the GROUP BY statement.
5) Pay attention to the columns that are used for the ORDER BY statement.
6) check that all of the columns exist in the table and there are no typos.
7) Use CAST when is needed.
8) USE CASE WHEN is needed.
###
Few examples of this task are:
###
Schema of the database with sample rows and column descriptions:
#
CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)

/*
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/

Table: movies
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi
#
Q: Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity.
Hint: released in the year 1945 refers to movie_release_year = 1945;
SQL: SELECT movie_title, movie_popularity FROM movies WHERE movie_release_year = 1945/01/01 ORDER BY movie_popularity DESC LIMIT 1
A: Let's think step by step to find the correct answer.
1) The column movie_popularity is not mentioned in the question so it's redundant.
2) JOIN is not required as there is no need to join any tables.
3) The condition movie_release_year = 1945/01/01 is not correct. The correct condition is movie_release_year = 1945.
4) GROUP BY is not required as there is no need to group any columns.
5) The ORDER BY clause is correct.
6) all columns are correct and there are no typo errors.
7) CAST is not required as there is no need to cast any columns.
8) CASE is not required as there is no need to use CASE.
So, the final sqlite SQL query answer to the question the given question is =
Revised_SQL: SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1
"""  # noqa: E501
HUMAN_SELF_CORRECTION_PROMPT = """
Evaluate the correctness of this query for the given question.
Hint helps you to write the correct SQL query.
Correct it if there are any issues. If there are no issues, return the SQLite SQL QUERY as is.
Schema of the database with sample rows and column descriptions:
#
{schema}

{columns_descriptions}
#
Q: {question}
Hint: {hint}
SQL: {sql_query}
A: Let's think step by step to find the correct answer.""" # noqa: E501

def get_database_schema(DB_URI: str) -> str:
    """Get the database schema from the database URI

    Args:
        DB_URI (str): Database URI

    Returns:
        str: Database schema
    """
    db = SQLDatabase.from_uri("sqlite:///"+DB_URI)
    db._sample_rows_in_table_info = 3
    return db.get_table_info_no_throw()

def extract_schema_links(input_text: str) -> List[str]:
    pattern = r'Schema_links:\s*\[(.*?)\]'
    match = re.search(pattern, input_text)
    if match:
        schema_links_str = match.group(1)
        schema_links = [link.strip() for link in schema_links_str.split(',')]
        return schema_links
    else:
        return []
    
def extract_label_and_sub_questions(input_text: str) -> Tuple[str, List[str]]:
    label_pattern = r'Label:\s*"(.*?)"'
    sub_questions_pattern = r'sub_questions:\s*\[(.*?)\]'

    label_match = re.search(label_pattern, input_text)
    sub_questions_match = re.search(sub_questions_pattern, input_text)

    label = label_match.group(1) if label_match else None

    sub_questions = []
    if sub_questions_match:
        sub_questions_str = sub_questions_match.group(1)
        sub_questions = [question.strip() for question in sub_questions_str.split(',')]

    return label, sub_questions

def extract_sql_query(input_text):
    sql_pattern = r'SQL:\s*(.*?)$'
    match = re.search(sql_pattern, input_text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_revised_sql_query(input_text):
    sql_pattern = r'Revised_SQL:\s*(.*?)$'
    match = re.search(sql_pattern, input_text, re.DOTALL)
    return match.group(1).strip() if match else None

def update_json_file(json_filename, index, sql_query, db_id):
    try:
        with open(json_filename, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}

    data[str(index)] = f"{sql_query}\t----- bird -----\t{db_id}"

    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def table_descriptions_parser(database_dir):
    csv_files = glob.glob(f"{database_dir}/*.csv")
    # Iterate over the CSV files
    db_descriptions = ""
    for file_path in csv_files:
        table_name: str = os.path.basename(file_path).replace(".csv", "")
        db_descriptions += f"Table: {table_name}\n"
        table_df = pd.read_csv(file_path, encoding='latin-1')
        for _,row in table_df.iterrows():
           try:
                if pd.notna(row[2]):
                    col_description = re.sub(r'\s+', ' ', str(row[2]))  # noqa: E501
                    val_description = re.sub(r'\s+', ' ', str(row[4]))
                    if pd.notna(row[4]):
                        db_descriptions += f"Column {row[0]}: column description -> {col_description}, value description -> {val_description}\n"  # noqa: E501
                    else:
                        db_descriptions += f"Column {row[0]}: column description -> {col_description}\n"  # noqa: E501
           except Exception as e:
                print(e)
                db_descriptions += "No column description"
        db_descriptions += "\n"
    return db_descriptions

start_index = 0

if __name__ == "__main__":
    logs_df = pd.DataFrame(
        columns=["question","gold_query","db_id","final_query","schema_linking","classification","sql_generation","self_correction"])

    # should be removed in the final version
    # dev_df = dev_df.sample(frac=0.03).reset_index(drop=True)
    # print("Number of questions: ", dev_df.shape[0])

    system_schema_linking_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_SCHEMA_LINKING_TEMPLATE)  # noqa: E501
    human_schema_linking_prompt = HumanMessagePromptTemplate.from_template(HUMAN_SCHEMA_LINKING_TEMPLATE) # noqa: E501
    schema_linking_prompt = ChatPromptTemplate.from_messages([system_schema_linking_prompt, human_schema_linking_prompt]) # noqa: E501
    system_classification_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_CLASSIFICATION_TEMPLATE) # noqa: E501
    human_classification_prompt = HumanMessagePromptTemplate.from_template(HUMAN_CLASSIFICATION_TEMPLATE) # noqa: E501
    classification_prompt = ChatPromptTemplate.from_messages([system_classification_prompt, human_classification_prompt]) # noqa: E501
    system_easy_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_EASY_CLASS_TEMPLATE) # noqa: E501
    human_easy_prompt = HumanMessagePromptTemplate.from_template(HUMAN_EASY_CLASS_TEMPLATE) # noqa: E501
    easy_prompt = ChatPromptTemplate.from_messages([system_easy_prompt, human_easy_prompt]) # noqa: E501
    system_easy_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_EASY_CLASS_TEMPLATE) # noqa: E501
    human_easy_prompt = HumanMessagePromptTemplate.from_template(HUMAN_EASY_CLASS_TEMPLATE) # noqa: E501
    easy_prompt = ChatPromptTemplate.from_messages([system_easy_prompt, human_easy_prompt]) # noqa: E501
    system_medium_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_NON_NESTED_CLASS_TEMPLATE) # noqa: E501
    human_medium_prompt = HumanMessagePromptTemplate.from_template(HUMAN_NON_NESTED_CLASS_TEMPLATE) # noqa: E501
    medium_prompt = ChatPromptTemplate.from_messages([system_medium_prompt, human_medium_prompt]) # noqa: E501
    system_hard_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_NESTED_CLASS_TEMPLATE) # noqa: E501
    human_hard_prompt = HumanMessagePromptTemplate.from_template(HUMAN_NESTED_CLASS_TEMPLATE) # noqa: E501
    hard_prompt = ChatPromptTemplate.from_messages([system_hard_prompt, human_hard_prompt]) # noqa: E501
    system_correction_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_SELF_CORRECTION_PROMPT) # noqa: E501
    human_correction_prompt = HumanMessagePromptTemplate.from_template(HUMAN_SELF_CORRECTION_PROMPT) # noqa: E501
    correction_prompt = ChatPromptTemplate.from_messages([system_correction_prompt, human_correction_prompt]) # noqa: E501
    accuracy = 0
    for index,row in dev_df.iterrows():
        if index < start_index:
            continue
        print("Processing row: ", index)
        db_uri = dev_db_path + "/" + row["db_id"] + "/" + row["db_id"] + ".sqlite"
        db_descriptions = dev_db_path + "/" + row["db_id"] + "/" + "database_description"  # noqa: E501
        print("Database: ", db_uri)
        columns_descriptions = table_descriptions_parser(db_descriptions)
        schema = get_database_schema(db_uri)
        question = row["question"]
        print("Question: ", question)
        hint = str(row["evidence"])
        question_id = row["question_id"]
        chain = LLMChain(llm=CHAT, prompt=schema_linking_prompt)
        schema_linking = chain.run(question=question, schema=schema, hint=hint, columns_descriptions=columns_descriptions) # noqa: E501
        schema_links = extract_schema_links(schema_linking)
        chain = LLMChain(llm=CHAT, prompt=classification_prompt)
        classification = chain.run(
            question=question,
            schema=schema,
            hint=hint,
            columns_descriptions=columns_descriptions,
            schema_links=schema_links)
        label, sub_questions = extract_label_and_sub_questions(classification)
        print("Label: ", label)
        sql_generation = None
        if "EASY" in label:
            chain = LLMChain(llm=CHAT, prompt=easy_prompt)
            easy = chain.run(
                question=question,
                schema=schema,
                hint=hint,
                columns_descriptions=columns_descriptions,
                schema_links=schema_links)
            sql_query = extract_sql_query(easy)
            sql_generation = easy
        elif "NON-NESTED" in label:     
            chain = LLMChain(llm=CHAT, prompt=medium_prompt)
            medium = chain.run(
                question=question,
                schema=schema,
                hint=hint,
                columns_descriptions=columns_descriptions,
                schema_links=schema_links)
            sql_query = extract_sql_query(medium)
            sql_generation = medium
        else:
            chain = LLMChain(llm=CHAT, prompt=hard_prompt)
            hard = chain.run(
                question=question,
                schema=schema,
                hint=hint,
                columns_descriptions=columns_descriptions,
                schema_links=schema_links,
                sub_questions=sub_questions)
            sql_query = extract_sql_query(hard)
            sql_generation = hard
        chain = LLMChain(llm=CHAT, prompt=correction_prompt)
        correction = chain.run(
            question=question,
            schema=schema,
            columns_descriptions=columns_descriptions,
            hint=hint,
            sql_query=sql_query)
        finall_sql = extract_revised_sql_query(correction)
        if finall_sql is not None:
            one_liner_sql_query = finall_sql.replace('\n', '').replace('\r', '')
        else:
            if sql_query is not None:
                one_liner_sql_query = sql_query.replace('\n', '').replace('\r', '')
            else:
                one_liner_sql_query = "SELECT * FROM table" # no query generated, placeholder to avoid errors # noqa: E501
        new_row_df = pd.DataFrame(
            [[question,row["SQL"],row["db_id"],one_liner_sql_query,schema_linking, classification, sql_generation, correction]],  # noqa: E501
            columns=["question","gold_query","db_id","final_query","schema_linking","classification","sql_generation","self_correction"])
        logs_df = pd.concat([logs_df, new_row_df], ignore_index=True)
        logs_df.to_csv("logs.csv", index=False)
        update_json_file("predict_dev.json", index, one_liner_sql_query, row["db_id"])
        print("final sql query: ", one_liner_sql_query)
        print("Gold sql query: ", row["SQL"])
        print("--------------------------------------------------")   