from dataload import *
from collaborative_recommender import Collaborative_Recommender
from latent_recommender import Matrix_Factorization_Recommender
from hybrid_recommender import Hybrid_Recommender
from tensor_recommender import Tensor_Recommender



dlc = DataLoader('reviews_100000.csv', 1000)
dlt = DataLoader('reviews_100000.csv', 10000)
dlm = DataLoader('reviews_100000.csv', 100000)


cbf_recs = Collaborative_Recommender(dlc.cbf_reviews_data(), dlc.ratings_data(), 30)
cbf_users = dlc.cbf_reviews_data().index.to_list()

file, user_dict, item_dict = dlm.spark_ratings()
mf_rec = Matrix_Factorization_Recommender(file, user_dict, item_dict)
mf_users = mf_rec.get_preds().index.to_list()

tf_recs = Tensor_Recommender(dlt.reviews_data(), dlt.ratings_data(), 30)

common_users = common_users_list(cbf_users, mf_users)

print('Number of Common users: {}\n'.format(len(common_users)))

#SAMPLE USER DETAILS - ID: 'AIDX2L5G4JD5D' NAME - EFJ

user = input('Enter your user ID: ')
username = dlc.get_username(user)
hyb_recs = Hybrid_Recommender(cbf_recs.recommendations(user), mf_rec.recommendations(user), tf_recs.recommendations(user))
print('Hi {}, these are the top 30 products that you may like: \n'.format(username))
print(hyb_recs.recommendations(user).head(30))
print('\n')
print('Algorithm-wise error evaluation: \n')
print('Collaborative Filtering: {}'.format(cbf_recs.get_error(user)))
print('Matrix Factorization: {}'.format(mf_rec.get_error(user)))
print('Tensor Factorization: {}'.format(tf_recs.get_error(user)))
print('Hybrid Algorithm: {}'.format(hyb_recs.get_error(user)))












