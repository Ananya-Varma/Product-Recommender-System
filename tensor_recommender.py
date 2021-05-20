import tensorly as tl
import pandas as pd
import numpy as np
from tensorly.decomposition import tucker
import math

class Tensor_Recommender:

    def __init__(self, data, user_item, num):
        self.data = data
        self.user_item = user_item
        self.num = num

    def get3D_matrix(self):
        df = self.data
        ratingsdf = df[['reviewerID','asin','overall']]
        ratingsdf = ratingsdf.pivot_table(values='overall',index='reviewerID',columns='asin').fillna(0)
        polaritydf = df[['reviewerID', 'asin', 'polarity']]
        polaritydf = polaritydf.pivot_table(values='polarity', index='reviewerID', columns='asin').fillna(0)
        subjectivitydf = df[['reviewerID', 'asin', 'subjectivity']]
        subjectivitydf = subjectivitydf.pivot_table(values='subjectivity', index='reviewerID', columns='asin').fillna(0)
        mat = np.dstack(ratingsdf,polaritydf,subjectivitydf)
        return ratingsdf.index, mat

    def latent_features(self):
        users, mat = self.get3D_matrix()
        tensor = tl.tensor(mat, dtype=tl.float64)
        tucker_tensor = tucker(tensor, rank=[360, 360, 1])
        return pd.DataFrame(tucker_tensor[1][0], index=users)

    def get_latent_matrix(self):
        df = pd.read_csv('latent.csv')
        return df.set_index('reviewerID')

    def top_n_similar_users(self, userID):
        users = self.get_latent_matrix()
        similar_to_user = users.transpose().corrwith(users.loc[userID])
        similar_to_user = pd.DataFrame(similar_to_user, columns=['Correlation'])
        similar_to_user = similar_to_user[similar_to_user['Correlation'] < 0.999]
        return similar_to_user.sort_values(by='Correlation', ascending=False)[1:self.num+1]

    def recommendations(self, user):
        user_item = self.user_item
        similar_users = self.top_n_similar_users(user)
        conf_df = pd.DataFrame(columns=['item', 'rating'])

        for users in similar_users.index:
            s = pd.DataFrame(user_item.loc[users])

            if s.columns.to_list()[0] == users:
                s = s.transpose()
                s['rating'] = s['rating'] * similar_users.loc[users]['Correlation']
                s = s.reset_index().drop('index', axis=1)
                conf_df = pd.concat([conf_df, s])
            else:
                s['rating'] = s['rating'] * similar_users.loc[users]['Correlation']
                s = s.reset_index().drop('user', axis=1)
                conf_df = pd.concat([conf_df, s])

        conf_df = conf_df.reset_index().drop('index', axis=1)
        conf_df['confidence'] = conf_df['rating'] * 20

        conf_df = conf_df.sort_values(by='confidence', ascending=False)[0:self.num].reset_index().drop(['rating', 'index'], axis=1)
        return conf_df

    def get_error(self, user):
        df = self.recommendations(user)
        df['rating'] = df['confidence']/20

        def round_off(x):
            floor = math.floor(x)

            if x - floor < 0.5:
                return floor
            else:
                return floor + 1

        df['actual'] = df['rating'].apply(lambda x: round_off(x))
        df['actual'] = df['actual'] - df['rating']
        df['actual'] = df['actual'].apply(lambda x: pow(x, 2))
        error = df['actual'].mean()
        return np.sqrt(error)





