import numpy as np

import pandas as pd

from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm import LightFM

import boto3

import os

import pickle

from time import sleep


def construct_dataset(df, user_id, bike_id, user_features, item_features):
    """construct the dataset including user and item features"""

    dataset = Dataset()

    features = user_features + item_features

    # convert all features to string
    for feature in features:
        df[feature] = df[feature].astype('str')

    # fit users, items
    dataset.fit(users=df[user_id].unique(), items=df[bike_id].unique())

    # flatten and convert to list all the user features from the dataframe
    user_features_flat = df[user_features].values.ravel('K').tolist()

    # add user features mapping to existing mappings
    dataset.fit_partial(users=df[user_id].unique(),
                        user_features=user_features_flat)

    # flatten and convert to list all the item features from the dataframe
    item_features_flat = df[item_features].values.ravel('K').tolist()

    # add item features mapping to existing mappings
    dataset.fit_partial(items=df[bike_id].unique(),
                        item_features=item_features_flat)

    return dataset


def construct_interactions(df, dataset, user_id, bike_id, feedback):
    """construct the interactions from the iterable [user_id, bike_id, feedback]


    Args:
        df (pd.DataFrame): dataframe
        dataset (lightfm.Dataset): lightfm dataset
        user_id (str): user id column name
        bike_id (str): bike id column name
        feedback (str): feedback column name
    Returns:
        interactions (lightfm.SparseMatrix): lightfm interactions
        interactions_weights (lightfm.SparseMatrix): lightfm interactions weights
        """

    interactions, interactions_weights = dataset.build_interactions(
        df[[user_id, bike_id, feedback]].values)

    return interactions, interactions_weights


def construct_item_features(df, dataset, bike_id, item_features):
    """construct the item features from the dataframe

    Args:
        df (pd.DataFrame): dataframe
        dataset (lightfm.Dataset): lightfm dataset
        bike_id (str): bike id column name
        item_features (list): list of item features column names
    Returns:
        item_features_matrix (lightfm.SparseMatrix): lightfm item features matrix

    """

    item_features_tuple_list = []

    for index, row in df.iterrows():
        feature_list = []
        for col in item_features:
            feature_value = row[col]
            feature_list.append(feature_value)
        item_features_tuple_list.append((row[bike_id], feature_list))

    item_features_matrix = dataset.build_item_features(
        item_features_tuple_list)

    return item_features_matrix


def construct_user_features(df, dataset, bike_id, user_features):
    """construct the user features from the dataframe

    Args:
        df (pd.DataFrame): dataframe
        dataset (lightfm.Dataset): lightfm dataset
        bike_id (str): bike id column name
        user_features (list): list of user features column names
    Returns:
        user_features_matrix (lightfm.SparseMatrix): lightfm user features matrix

    """

    user_features_tuple_list = []

    for index, row in df.iterrows():
        feature_list = []
        for col in user_features:
            feature_value = row[col]
            feature_list.append(feature_value)
        user_features_tuple_list.append((row[bike_id], feature_list))

    user_features_matrix = dataset.build_user_features(
        user_features_tuple_list)

    return user_features_matrix


def construct_train_test(interactions, interactions_weights, test_percentage=0.2):

    train, test = random_train_test_split(
        interactions, test_percentage=test_percentage, random_state=np.random.RandomState(3))

    train_weights, test_weights = random_train_test_split(
        interactions_weights, test_percentage=test_percentage, random_state=np.random.RandomState(3))

    return train, train_weights, test, test_weights


def construct_model(train,
                    user_features_matrix,
                    item_features_matrix,
                    weights,
                    epochs=10,
                    num_components=30,
                    learning_rate=0.05,
                    loss='warp',
                    random_state=1):
    """Initialize a LightFM model instance and fit to the training data
    Args:
        train (lightfm.SparseMatrix): lightfm training interactions
        user_features_matrix (lightfm.SparseMatrix): lightfm user features matrix
        item_features_matrix (lightfm.SparseMatrix): lightfm item features matrix
    Returns:
        model (lightfm.LightFM): lightfm model instance
    """

    model = LightFM(learning_rate=learning_rate, loss=loss,
                    no_components=num_components, random_state=random_state)

    model.fit(train,
              user_features=user_features_matrix,
              item_features=item_features_matrix,
              sample_weight=weights,
              epochs=epochs, num_threads=4)

    return model


def get_model(df,
              user_id,
              bike_id,
              user_features,
              item_features,
              feedback='feedback',
              k=4,
              test_percentage=0.2,
              epochs=10,
              num_components=30,
              learning_rate=0.05,
              loss='warp',
              random_state=1):
    """construct necessary datasets and fit model

    Args:
        df (pd.DataFrame): dataframe
        user_id (str): user id column name
        bike_id (str): bike id column name
        user_features (list): list of user features column names
        item_features (list): list of item features column names
    Returns:
        model (lightfm.LightFM): lightfm model instance
        train (lightfm.SparseMatrix): lightfm training interactions
        test (lightfm.SparseMatrix): lightfm test interactions
        dataset (lightfm.Dataset): lightfm dataset
        interactions (lightfm.SparseMatrix): lightfm interactions
        interactions_weights (lightfm.SparseMatrix): lightfm interactions weights
        item_features_matrix (lightfm.SparseMatrix): lightfm item features matrix

     """

    # xxxxx get the flow of data right and check where adjustments are necessary
    # make the bike_id column an integer
    df[bike_id] = df[bike_id].astype(int)

    dataset = construct_dataset(
        df, user_id, bike_id, user_features, item_features)

    interactions, interactions_weights = construct_interactions(
        df, dataset, user_id, bike_id, feedback)

    train, train_weights, test, test_weights = construct_train_test(
        interactions, interactions_weights, test_percentage)

    user_features_matrix = construct_user_features(
        df, dataset, user_id, user_features)

    item_features_matrix = construct_item_features(
        df, dataset, bike_id, item_features)

    model = construct_model(train,
                            user_features_matrix,
                            item_features_matrix,
                            train_weights,
                            epochs,
                            num_components,
                            learning_rate,
                            loss,
                            random_state)

    return model, train, test, dataset, interactions, interactions_weights, user_features_matrix, item_features_matrix


def update_model(df,
                 user_id,
                 bike_id,
                 user_features,
                 item_features,
                 path):
    """ retrain and write model to disk"""
    model, train, test, dataset, interactions, interactions_weights, user_features_matrix, item_features_matrix = get_model(
        df, user_id, bike_id, user_features, item_features
    )

    write_model_data(model, dataset, path)


def auc(model, train, test, user_features_matrix, item_features_matrix, num_threads=4):
    """calculate auc score for train and test data"""
    train_auc = auc_score(
        model, train, user_features=user_features_matrix, item_features=item_features_matrix, num_threads=num_threads).mean()

    test_auc = auc_score(
        model, test, user_features=user_features_matrix, item_features=item_features_matrix, num_threads=num_threads).mean()

    return train_auc, test_auc


def eval_model(model, train, test, user_features_matrix, item_features_matrix, k=4, num_threads=4):
    """calculate precision, train auc and test auc for model"""
    precision = precision_at_k(
        model=model,
        test_interactions=test,
        train_interactions=train,
        item_features=item_features_matrix,
        user_features=user_features_matrix,
        k=k,
        num_threads=num_threads,
        check_intersections=False).mean()

    train_auc, test_auc = auc(
        model,
        train,
        test,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        num_threads=num_threads)

    return precision, train_auc, test_auc


def write_model_data(model, dataset, path):
    with open(path + "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(path + "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


def read_data_model(path="data/"):
    with open(path + "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(path + "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    return model, dataset


def get_top_n_collaborative(model, user_id: str, n: int, dataset, logger):
    """
    Retrieve the top k item ids for a given user_id by using model.predict()

    Args:
        model (LightFM): Trained LightFM model.
        user_id (str): user_id for which to retrieve top k items.
        n (int): Number of top items to retrieve.
        dataset (Dataset): LightFM dataset object containing mapping between internal and external ids.
        logger (Logger): Logger object.

    Returns:
        list: List of top n item ids for the given user.
        str: Error message if any.
    """
    error = None
    top_n_item_ids = []

    try:
        # map user_id to user_id in dataset
        user_id_index = dataset.mapping()[0][user_id]

        n_items = dataset.interactions_shape()[1]
        item_ids = np.arange(n_items)
        scores = model.predict(user_id_index, item_ids)
        top_n_items = np.argsort(-scores)[:n]

        # Map internal item index back to external item ids
        item_index_id_map = {v: n for n, v in dataset.mapping()[2].items()}

        top_n_item_ids = [item_index_id_map[item_id]
                          for item_id in top_n_items]

        return top_n_item_ids, error

    except Exception as e:
        error = str(e)
        return top_n_item_ids, error


def create_data_model_collaborative(s3_client,
                                    s3_credentials,
                                    get_last_date_S3,
                                    get_last_date_local,
                                    get_multi_day_data,
                                    implicit_feedback,
                                    features,
                                    user_features,
                                    item_features,
                                    user_id,
                                    bike_id,
                                    combine_data,
                                    write_data,
                                    update_model,
                                    path: str = "data/",
                                    limit=2,
                                    fraction=0.8):
    """ create data model for collaborative filtering
        if S3 has newer data than local, update local data
        retrain and write model (update) and write dataset to path if new data was available or model.pkl is missing
    Args:
        path (str): path to data
        limit (int): the number of days if there is no local data available
    """

    Bucket, Key, Filename = s3_credentials()

    client = s3_client()
    if os.path.isfile(path + "df_collaborative.pkl"):
        df = pd.read_pickle(path + "df_collaborative.pkl")
    else:
        df = None
        metadata = None

    # check if there is newer data than the metadata of df_collaborative
    # if yes, update df_collaborative
    # if no, use df_collaborative

    if get_last_date_S3(client, Bucket) > get_last_date_local(path, limit):

        df_new, metadata = get_multi_day_data(
            start_date=get_last_date_local(path, limit),
            end_date=get_last_date_S3(client, Bucket),
            implicit_feedback=implicit_feedback,
            features=features,
            user_features=user_features,
            item_features=item_features,
            user_id=user_id,
            bike_id=bike_id,
            Bucket=Bucket,
            inclusive='right',
            fraction=fraction,)

        df = combine_data(df, df_new, user_id, bike_id,
                          user_features, item_features)

        write_data(df, metadata, path)

        update_model(df,
                     user_id,
                     bike_id,
                     user_features,
                     item_features,
                     path)

    # if the model exists, do nothing, else create model
    if os.path.isfile(path + "model.pkl"):
        pass
    else:
        update_model(df,
                     user_id,
                     bike_id,
                     user_features,
                     item_features,
                     path)


class DataStoreCollaborative:
    def __init__(self):
        self.model = None
        self.dataset = None

    def read_data_periodically(self, period, logger):
        """Read data periodically
        Args:
            period: period in minutes
            logger: logger
        """
        error = None
        period = period * 60

        while True:
            try:
                self.model, self.dataset = read_data_model()
                logger.info("Data and model read, Collaborative")

            except Exception as error:
                logger.error("Data could not be read: " + str(error))
            sleep(period)


###################### dev##########################################################


# deal with ex post updated distinct_id
# check the whole distinct_id user_id pairs where user_id is not null

# check preferneces df where user_id is null
# merge the preferneces values of this distinct_id with the user_id from the mapping

# download S3 distinct_id user_id mapping
def merge_distinct_id(df, mapping):
    pass


# collaborative filtering

# given a feeback matrix m x n (user_id x bike_id)
