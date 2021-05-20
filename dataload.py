import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from textblob import TextBlob
from sklearn.cluster import KMeans

class DataLoader:

    def __init__(self, file, nrows):
        self.df = pd.read_csv(file, nrows=nrows)
        self.size = nrows

    def get_username(self, userID):
        df = self.df[['reviewerID', 'reviewerName']]
        name = df[df['reviewerID'] == userID]['reviewerName'].to_list()
        return name[0]

    def fake_review_correction(self):
        df = self.df[['reviewerID', 'asin', 'overall', 'reviewText']]
        df['polarity'] = df['reviewText'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        df['subjectivity'] = df['reviewText'].astype(str).apply(lambda x: TextBlob(x).sentiment.subjectivity)
        df.drop('reviewText', axis=1, inplace=True)
        df = df.groupby(['reviewerID', 'asin']).mean()

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(df)
        labels = pd.DataFrame(kmeans.labels_, columns=['labels'])
        df = df.reset_index()
        df['labels'] = labels['labels']

        items_df = df[['asin', 'labels']]
        items_df = df.groupby('asin').mean()
        items_df = items_df.reset_index()

        item_dict = dict(zip(items_df['asin'], items_df['labels']))

        def correcter(asin, item_dict):
            for key,value in item_dict.items():
                if key == asin:
                    return value

        df['correction'] = df['asin'].apply(lambda x: correcter(x, item_dict))
        df['overall'] = df['overall'] - df['correction']
        df.drop(['correction','labels'], axis=1, inplace=True)
        return df

    def reviews_data(self):
        df = self.df
        df = df.groupby(['reviewerID', 'asin']).mean().reset_index()
        return df

    def ratings_data(self):
        df = self.df[['reviewerID', 'asin', 'overall']]
        df = df.groupby(['reviewerID', 'asin']).mean().reset_index()
        df['user'] = df['reviewerID']
        df['item'] = df['asin']
        df['rating'] = df['overall']
        df = df.drop(['reviewerID','asin', 'overall'], axis=1)
        return df.set_index('user')

    def cbf_reviews_data(self):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(self.df['summary'].astype(str))
        features = vectorizer.get_feature_names()
        dense_list = vectors.todense()
        dense = pd.DataFrame(dense_list,columns=features)

        scaler = StandardScaler()
        scaler.fit(dense)
        scaled_data = scaler.transform(dense)
        pca = PCA(n_components=0.8, svd_solver='full')
        pca.fit(scaled_data)
        pdf = pca.transform(dense)
        pdf = pd.DataFrame(pdf)

        pdf['user'] = self.df['reviewerID']
        pdf['item'] = self.df['asin']
        pdf['rating'] = self.df['overall']

        pdf['polarity'] = self.df['reviewText'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        pdf['subjectivity'] = self.df['reviewText'].astype(str).apply(lambda x: TextBlob(x).sentiment.subjectivity)

        pdf = pdf.groupby(['user', 'item']).mean().reset_index()
        return pdf.groupby('user').mean()

    def spark_ratings(self):
        df = self.df[['reviewerID', 'asin', 'overall']]
        df = df.groupby(['reviewerID', 'asin']).mean().reset_index()
        users = df['reviewerID'].unique()
        items = df['asin'].unique()
        user_dict = dict(enumerate(users, 1))
        item_dict = dict(enumerate(items, 1))
        sdf = pd.DataFrame(df)

        def get_key(val,dc):
            for key, value in dc.items():
                if val == value:
                    return key

        sdf['user'] = sdf['reviewerID'].astype(str).apply(lambda x: get_key(x, user_dict))
        sdf['item'] = sdf['asin'].astype(str).apply(lambda x: get_key(x, item_dict))
        sdf['rating'] = sdf['overall']
        sdf = sdf.drop(['reviewerID', 'asin', 'overall'], axis=1)
        filename = 'spark_ratings.csv'
        sdf.to_csv(filename, index=False)
        return filename, user_dict, item_dict


def common_users_list(ls1, ls2):
    set1 = set(ls1)
    set2 = set(ls2)
    common = set.intersection(set1, set2)
    common = list(common)
    return common

def get_itemname(itemID):
    df = pd.read_csv('metadata.csv', nrows=150000)
    sp = df[df['asin'] == itemID]

    if len(sp) != 0:
        return sp['title']

    else:
        return itemID