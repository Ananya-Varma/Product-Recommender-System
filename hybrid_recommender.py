import pandas as pd
import numpy as np
import math


class Hybrid_Recommender:

    def __init__(self, cbf_recs, mf_recs, tf_recs):
        self.recs1 = cbf_recs
        self.recs2 = mf_recs
        self.recs3 = tf_recs

    def recommendations(self, user):
        df = pd.concat([self.recs1, self.recs2, self.recs3], axis=0)
        df = df.sort_values(by='confidence', ascending=False).reset_index().drop('index', axis=1)
        return df

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