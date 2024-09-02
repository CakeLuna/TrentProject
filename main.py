import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)

df = pd.read_csv("Products_Report (5).csv")
df2 = df[["ID","Title", "Tags","Variant SKU"]]
df2 = df2.drop_duplicates(subset='Title')
df2.dropna(inplace=True)
df2 = df2[:1000]
df2["Tags"] = df2["Tags"].apply(lambda x: x.split(","))
df2["Tags"]= df2["Tags"].apply(lambda x: [item for item in x if item not in [' Exclude', ' offrange_delete', ' No Returns']])

df2["Tags"] = df2["Tags"].apply(lambda x: ','.join(x))


def remove_numbers_from_tags(tags):
  """Removes values containing numbers from a string of tags."""
  tags_list = tags.split(',')
  filtered_tags = [tag for tag in tags_list if not re.search(r'\d', tag)]
  return ','.join(filtered_tags)

df2["Tags"] = df2["Tags"].apply(remove_numbers_from_tags)


def boring_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()

    words = [word for word in words if word!='']
    return words

count_vectorizer = CountVectorizer(stop_words='english',tokenizer = boring_tokenizer)
X = count_vectorizer.fit_transform(df2["Tags"])

cosine_sim = cosine_similarity(X,X)



# prompt: I want to apply cosine similarity on tags and df2["tags"]

def recommend_products(sku):
  """
  Recommends products based on cosine similarity of tags.
  """
  try:
    # Get the index of the product with the given SKU
    index = df2[df2["Variant SKU"] == sku].index[0]
  except IndexError:
    return "SKU not found."

  # Get the similarity scores for the given product
  similarity_scores = list(enumerate(cosine_sim[index]))

  # Sort the products based on similarity scores
  similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

  # Get the top 10 most similar products
  similarity_scores = similarity_scores[1:11]

  # Get the indices of the recommended products
  product_indices = [i[0] for i in similarity_scores]

  # Return the recommended products
  return df2[["Variant SKU", "Title"]].iloc[product_indices]

# Example usage:
# sku = df2["Variant SKU"][900] 
# recommendations = recommend_products(sku)
# print(f"Recommendations for SKU {sku}:\n{recommendations}")

@app.route('/rec/<int:sku>',methods=['GET'])
def get_rec(sku):
  rec_dict = recommend_products(sku).to_dict(orient='records')
  return jsonify(rec_dict)

if __name__ == '__main__':
  app.run(debug=True)

