from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import jieba


df = pd.read_excel(r'D:\Thesis\Data\ModelData\tomato_data.xlsx')[['Phrase','Sentiment']].rename(columns={'Phrase':'text','Sentiment':'label'})

stored_reviews = df['text'].to_list()

custom_attribute_words = ["theme","plot","role", "soundtrack", "screen"]


initial_keywords = [
    ['science fiction', 'disaster', 'imagination', 'mars', 'moon', 'marvel', 'dc'],
    ['story', 'original work', 'logic', 'plot', 'pacing', 'ending', 'creativity'],
    ['actors', 'original work', 'leading role', 'robert downey jr', 'cameo', 'characters', 'batman'],
    ['sound effects', 'music', 'pace', 'lip-sync', 'subtitles', 'sound', 'high technology'],
    ['imax', 'scenes', 'cinematography', '3d', 'special effects', 'technology', 'editing']
]

tokenized_reviews = []
for review in stored_reviews:
    tokens = list(jieba.cut(review))
    tokens = [token for token in tokens if token]
    tokenized_reviews.append(tokens)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stored_reviews)
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X)

word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
similar_words = {}
for word_list in initial_keywords:
    for word in word_list:
        if word in word2vec_model.wv.key_to_index:
            similar_words[word] = [w[0] for w in word2vec_model.wv.most_similar(word) if w[1] > 0.8]

topic_attribute_set = {}
for i, topic_weights in enumerate(lda_model.components_):
    topic_keywords = [vectorizer.get_feature_names()[word_id] for word_id in topic_weights.argsort()[:-6:-1]]
    initial_keywords_per_topic = [similar_words[word] for word in initial_keywords[i] if word in similar_words]
    combined_words = set(topic_keywords + custom_attribute_words + initial_keywords[i] + [item for sublist in initial_keywords_per_topic for item in sublist])
    topic_attribute_set[i] = combined_words

topic_reviews = {topic: [] for topic in range(5)}
for review in stored_reviews:
    for topic, attributes in topic_attribute_set.items():
        if any(attr in review for attr in attributes):
            topic_reviews[topic].append(review)

data = []
for key, values in topic_reviews.items():
    for value in values:
        data.append((key, value))

topic_reviews_df = pd.DataFrame(data, columns=['topic', 'text'])

topic_reviews_res_df = pd.merge(topic_reviews_df,df,how='left',on='text')


topic_label_counts = topic_reviews_res_df.groupby(['topic', 'label']).size().unstack(fill_value=0)
print("every topic's label number：")
print(topic_label_counts)

topic_totals = topic_label_counts.sum(axis=1)
print("\ntotal number of every topic：")
print(topic_totals)

topic_label_ratios = topic_label_counts.div(topic_totals, axis=0)
print("\nevery topic's label percentage：")
print(topic_label_ratios)