import pandas as pd 

movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")



#Preprocessing data deleting columns 
#Remove the time timestamp column from ratings.csv
#Remove the time timestamp column from movies.csv
movies_df = movies_df.drop(["genres"], axis = 1)
ratings_df = ratings_df.drop(["timestamp"], axis = 1)
# print(movies_df.head(5))
# print(ratings_df.head(5))
merged_df = pd.merge(movies_df, ratings_df, on="movieId")
grouped_by = merged_df.groupby (["userId", "title"])["rating"].mean()
print(grouped_by.head(5))
# pivot_table = merged_df.pivot(index ="userId", columns="title", values="rating")
# # # print(movies_df.head(5))
# # # print(ratings_df.head(5))
# print(pivot_table.head(5))