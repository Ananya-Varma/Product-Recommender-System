import pandas as pd
import os
spark_home = os.environ['SPARK_HOME']
import findspark
findspark.init(spark_home=spark_home)
from pyspark import SparkContext
sc = SparkContext()
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import numpy as np
import math

class Matrix_Factorization_Recommender:

    def __init__(self, file, user_dict, item_dict):
        self.ratings = file
        self.user_dict = user_dict
        self.item_dict = item_dict

    def spark_dataload(self):
        spdf = self.ratings
        spdf.show()
        spdf = spdf.selectExpr('cast(user as int)user', 'cast(item as int)item', 'cast(rating as float)rating')
        return spdf

    def training_model(self):
        spdf = self.spark_dataload()
        X_train, X_test = spdf.randomSplit([0.7,0.3], seed=0)
        als = ALS(rank=20, maxIter=20, regParam=.18, seed=0, coldStartStrategy='drop', nonnegative=True)
        model = als.fit(X_train.select(['user', 'item', 'rating']))

        predictions = model.transform(X_test.select(['user', 'item']))

        ratesAndPreds = X_test.join(predictions, (X_test.user == predictions.user) &
                                    (X_test.item == predictions.item), how='inner').select(X_test.user, X_test.item, X_test.rating, predictions.prediction)

        ratesAndPreds = ratesAndPreds.select([col('rating').alias('label'), col('prediction').alias('raw')])

        ratesAndPreds = ratesAndPreds.withColumn('label', ratesAndPreds['label'].cast(FloatType()))

        return predictions, ratesAndPreds

    def error_evaluation(self):
        predictions, ratesAndPreds = self.training_model()
        evaluator = RegressionEvaluator(predictionCol='raw')
        return evaluator.evaluate(ratesAndPreds, {evaluator.metricName: 'rmse'})

    def get_predictions(self):
        predictions, ratesAndPreds = self.training_model()
        predictions.write.csv('preds.csv')
        pred = pd.read_csv('preds.csv')

        def get_val(k,dc):
            for key,value in dc.items():
                if key == k:
                    return value

        pred['user'] = pred['user'].apply(lambda x: get_val(x, self.user_dict))
        pred['item'] = pred['item'].apply(lambda x: get_val(x, self.item_dict))

        return pred

    def get_preds(self):
        pred = pd.read_csv('matrix_factor.csv')
        pred['confidence'] = pred['confidence']
        pred = pred.set_index('user')
        return pred

    def recommendations(self,user):
        pdf = self.get_preds()
        #pdf['confidence'] = pdf['confidence'] * 5 / pdf['confidence'].max()
        #pdf = pdf.drop('rating', axis=1)
        #pdf = pdf.set_index('user')

        items_df = pd.DataFrame(pdf.loc[user])

        if items_df.columns.to_list()[0] == user:
            conf_df = pd.DataFrame(items_df).transpose()
            conf_df = conf_df.reset_index()
            conf_df.drop('index', axis=1, inplace=True)
        else:
            conf_df = pd.DataFrame(items_df)
            conf_df = conf_df.reset_index()
            conf_df.drop('user', axis=1, inplace=True)

        conf_df['confidence'] = conf_df['confidence'] * 20

        return conf_df.sort_values(by='confidence', ascending=False).reset_index().drop('index', axis=1)

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