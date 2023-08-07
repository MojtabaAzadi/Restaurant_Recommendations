import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("TA_restaurants_curated.csv")

data = data[["Name", "Cuisine Style", "City"]]

london_selection = data[data["City"] == "London"]

london_selection = london_selection.dropna()

london_selection.index = london_selection.index - 45580

feature = london_selection["Cuisine Style"].tolist()
tfidf = text.TfidfVectorizer(input="content", stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(london_selection.index, index=london_selection['Name']).drop_duplicates()


def recommendations(name, similarity=similarity):
    index = indices[name]
    similarity_rate = list(enumerate(similarity[index]))
    similarity_rate = sorted(similarity_rate, key=lambda x: x[1], reverse=True)
    similarity_rate = similarity_rate[0:10]
    restaurant_indices = [i[0] for i in similarity_rate]
    return london_selection['Name'].iloc[restaurant_indices]


customer_restaurant_history = input("Name of the restaurant the customer has visited in the past:")

print("Suggested Restaurants:")
print(recommendations(customer_restaurant_history))