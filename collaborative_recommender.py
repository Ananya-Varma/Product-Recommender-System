import pandas as pd
import numpy as np
import math


class Collaborative_Recommender:

    def __init__(self, dat1, dat2, num):
        self.mat = dat1
        self.user_item = dat2
        self.num = num

    def matrix_generator(self):
        user_mat = self.mat
        return user_mat

    def top_n_similar_users(self, user):
        user_mat = self.matrix_generator()
        user_feature = user_mat.loc[user]
        similar_to_user = user_mat.transpose().corrwith(user_feature)
        similar_to_user = pd.DataFrame(similar_to_user,columns=['Correlation'])
        return similar_to_user.sort_values(by='Correlation',ascending=False).reset_index()[1:self.num+1].set_index('user')

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
        conf_df = conf_df[conf_df['confidence'] < 99]
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
        df['actual'] = df['actual'].apply(lambda x: pow(x,2))
        error = df['actual'].mean()
        return np.sqrt(error)