import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")

#Preprocessing data deleting columns 
   #Remove the time timestamp column from ratings.csv
   #Remove the time timestamp column from movies.csv
movies_df = movies_df.drop(["genres"], axis = 1)
ratings_df = ratings_df.drop(["timestamp"], axis = 1)
# print(movies_df.head(5))
# print(ratings_df.head(5))
data = pd.merge(movies_df, ratings_df, on="movieId")
#rating has multiple value so we had to get the mean 
data = data.groupby (["userId", "title"])["rating"].mean()
#Index is reset so that userId, title and rating are all columns.
data = data.reset_index() 

#Create user-item matrix
user_title_matrix = data.pivot(index="userId", columns="title", values="rating")

#Replace NaN values with zeros using Pandas fillna()
data_fillna = user_title_matrix.fillna(0)
# print(data.head(5))

#Compute similarity between users using cosine similarity 
   #Import cosine_similarity from skleab.metrics.pairwise 
   #Pass value into cosine_similarity method
data_cos_sim = cosine_similarity(data_fillna)
# print(data_cos_sim)
data = pd.DataFrame(data_cos_sim)
# print(data.head(5))

# 4 — BUILDING RECOMMENDATION FUNCTION 
# 4.1 — Get index of taget user we will be generating recommendations for. 
target_user_index = 1
# 4.2 — Retrieve similarity score between target user and alll other users
   # .loc() lets us pick specific  rows in dataframe 
   # Filter out the target index because it would bias the recommendation 
   # We only want to base recommendations on what other similar users have like not what the user already rated.
   # so we need to drop the row of the user

all_rows = data.loc[target_user_index]
similar_rows = all_rows.drop(target_user_index)
# 4.3 — Users sorted by similarity score(descending order)
   # We dont want all users we want only the most similar 
   # grab top 5 users
similar_rows = pd.DataFrame(similar_rows)
sorted_desc = similar_rows.sort_values(by=1, ascending=False)
top_five_similar = sorted_desc.nlargest(5, 1)
# 4.4 — Retrieve Movie rating of Top Similar Users
   # These are the users whose ratings we'll analyze to find movies our target user has no seen but might like 
similar_user_rating = data_fillna.loc[top_five_similar.index]
# print(similar_user_rating)

# 4.5 - Identify movies the target user has not rated.
   #First we get only the rows of movies of all movies rated and unrated by the user 
   #Filter out movies that are not rated 

target_user_row = data_fillna.loc[target_user_index]
target_user_unrated_movies = target_user_row[target_user_row == 0.0]

#4.6 - Score the candidate movies using weighted similarity: 
   # This step involves predicting how much the target user would like each unseen movie based on how similar users rated it rated it 
   
   # WHAT TO DO ? 
   #1. For each movie not rated by user Check how the top 5 similar users rated it 
   #2. For each rating, multiply it by that user's similarity score
   #3. sum the weighted scores across all similar users 
   #4. divide by the same of similarity scores for users who rated the movie (Giving a weighted average rating per movie)

top_five_similar_rating = data_fillna.loc[top_five_similar.index, target_user_unrated_movies.index]
# for i in top_five_similar_rating.index:
#    each_rating = top_five_similar_rating.loc[i]

#    print(each_rating)
print(top_five_similar_rating.loc[365])


# print(data_fillna)
# print(top_five_similar)
#choose user you want to

# pivot_table = merged_df.pivot(index ="userId", columns="title", values="rating")
# # # print(movies_df.head(5))
# # # print(ratings_df.head(5))
# print(pivot_table.head(5))