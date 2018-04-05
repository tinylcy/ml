import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

HOUSING_PATH = './datasets'
HOUSING_URL = HOUSING_PATH + '/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	housing_tgz = tarfile.open(housing_url)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()


# Returns a Pandas DataFrame object containing all the data.
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, 'housing.csv')
	return pd.read_csv(csv_path)


# Take a Quick Look at the Data Structure.
# housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# ################### Create a Test Set ###################

# def test_set_check(identifier, test_ratio, hash):
# 	return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
#
#
# def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
# 	ids = data[id_column]
# 	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
# 	return data.loc[~in_test_set], data.loc[in_test_set]
#
#
# def split_train_test(data, test_ratio):
# 	shuffled_indices = np.random.permutation(len(data))
# 	test_set_size = int(len(data) * test_ratio)
# 	test_indices = shuffled_indices[:test_set_size]
# 	train_indices = shuffled_indices[test_set_size:]
# 	return data.iloc[train_indices], data.iloc[test_indices]
#
#
# housing = load_housing_data()
# housing_with_id = housing.reset_index()  # Add an `index` column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
# print(len(train_set), 'train +', len(test_set), 'test')

# Using split_train_test function provided by Scikit-Learn to split datasets.
housing = load_housing_data()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set), 'train +', len(test_set), 'test')
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
# Do stratified sampling based on the income category.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

# print(housing['income_cat'].value_counts() / len(housing))
# Remove the income_cat attribute so the data is back to its original state.
for set in (strat_train_set, strat_test_set):
	set.drop(['income_cat'], axis=1, inplace=True)

# ################### Discover and visualize the data to gain insights. ###################

# Create a copy so I can play with the training set without harming.
housing = strat_train_set.copy()

# Is is a good idea to create a scatterplot of all districts to visualize the data.
# With alpha=0.1 I can clearly see the high-density areas.
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# plt.show()

# Now let's look at the housing prices.
# The radius of each circle represents the district's population (option s), and the color
# represents the price (option c).
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population'] / 100,
# 			 label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()
# plt.show()

# Look for correlations.
# Since the dataset is not too large, I can easily compute the standard correlation coefficient between every
# pair of attributes.
# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

# Another way to check for correlation between attributes is to use Pandas'scatter_matrix function,
# which plots every numerical attribute against every other numerical attribute.
# attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# The most promising attribute to predict the median house value is the median income.
# housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
# plt.show()

# Experiment with attribute combinations.
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
