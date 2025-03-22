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
data = data.pivot(index="userId", columns="title", values="rating")

#Replace NaN values with zeros using Pandas fillna()
data = data.fillna(0)
# print(data.head(5))

#Compute similarity between users using cosine similarity 
   #Import cosine_similarity from skleab.metrics.pairwise 
   #Pass value into cosine_similarity method
data_cos_sim = cosine_similarity(data)
# print(data_cos_sim)
data = pd.DataFrame(data_cos_sim, index = ["userId", "userId"])
print(data)

# pivot_table = merged_df.pivot(index ="userId", columns="title", values="rating")
# # # print(movies_df.head(5))
# # # print(ratings_df.head(5))
# print(pivot_table.head(5))