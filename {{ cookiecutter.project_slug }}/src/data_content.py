import pandas as pd
import numpy as np

# config file
import configparser

# DB connection
from sqlalchemy import create_engine, text  # database connection

# feature engineering
from sklearn.preprocessing import MinMaxScaler


# similarity estimation
from sklearn.metrics import pairwise_distances_chunked
from scipy.spatial.distance import cdist  # , pdist, squareform

from time import sleep


def frame_size_code_to_numeric(df: pd.DataFrame, bike_type_id_column="bike_type_id", frame_size_code_column="frame_size_code") -> pd.DataFrame:
    """map string frame_size_code with numeric frame_size_code
    Args:
        df (pandas.DataFrame): dataframe of bikes
        bike_type_id_column (str): column name of bike_type_id
        frame_size_code_column (str): column name of frame_size_code

    Returns:
        df (pandas.DataFrame): dataframe of bikes with numeric frame_size_code
    """
    # for each bike_type_id replace the frame_size_code with a numiric value from a dictionery
    frame_size_code_to_cm = {
        1:  {
            "xxs": "46",
            "xs": "49",
            "s": "52",
            "m": "54",
            "l": "57",
            "xl": "60",
            "xxl": "62",
            "xxxl": "64",
        },
        2:  {
            "xxs": "33",
            "xs": "36",
            "s": "41",
            "m": "46",
            "l": "52",
            "xl": "56",
            "xxl": "58",
            "xxxl": "61",
        },
    }

    # Filter dataframe to only include rows where frame_size_code is in the dictionary for the given bike_type_id and is non-numeric
    mask = (
        df[frame_size_code_column].isin(
            ["xxs", "xs", "s", "m", "l", "xl", "xxl", "xxxl"]
        )
        & df[frame_size_code_column].str.isnumeric().eq(False)
    )

    # Replace the frame_size_code with the numeric value from the dictionary
    df.loc[mask, frame_size_code_column] = df.loc[mask].apply(
        lambda row: frame_size_code_to_cm[row[bike_type_id_column]][row[frame_size_code_column]], axis=1
    )

    # Transform the frame_size_code to numeric, for the already numeric but in string format
    df[frame_size_code_column] = pd.to_numeric(df[frame_size_code_column])

    return df

# get the ids with a certain status


def get_data_status_mask(df: pd.DataFrame, status: list) -> pd.DataFrame:
    """get the ids with a certain status
    Args:
        df (pandas.DataFrame): dataframe of bikes
        status (list): list of status to filter by

    Returns:
        df (pandas.DataFrame): dataframe of bikes with the given status
    """

    mask = df.index[df["status"].isin(status)].tolist()

    return mask


def feature_engineering(
    df: pd.DataFrame,
    categorical_features: list,
    categorical_features_to_overweight: list,
    categorical_features_overweight_factor: float,
    numerical_features: list,
    numerical_features_to_overweight: list,
    numerical_features_overweight_factor: float,

) -> pd.DataFrame:
    """feature engineering for the bike dataframe,
    only keeps the categorical and numerical features
    one hot encodes the categorical features, devides by the square root of the unique number of unique categories
    min max scales the numerical features and reweights them according to the ratio of categorical to numerical features
    overweighing of categorical and numerical features available
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (list): list of categorical features
        categorical_features_to_overweight (list):
        categorical_features_overweight_factor (float):
        numerical_features (list): list of numerical features
        numerical_features_to_overweight (list)
        numerical_features_overweight_factor (float)

    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """

    df = df[categorical_features + numerical_features]
    df = categorical_encoding(df,
                              categorical_features,
                              categorical_features_to_overweight,
                              categorical_features_overweight_factor)
    df = numerical_scaling(df,
                           categorical_features,
                           numerical_features,
                           numerical_features_to_overweight,
                           numerical_features_overweight_factor)

    return df


def categorical_encoding(df: pd.DataFrame, categorical_features: list, categorical_features_to_overweight: list, categorical_features_overweight_factor: float) -> pd.DataFrame:
    """categorical encoding for the bike dataframe
    dummy variable encode and reweight according to number of unique values
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (list): list of categorical features
        categorical_features_to_overweight (list): list of categorical features to overweight
        categorical_features_overweight_factor (float): factor to overweight the categorical features by
    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """

    # Get the number of unique values for each categorical column
    unique_values_dict = {
        column: len(df[column].unique()) for column in categorical_features
    }

    # one hot encode the categorical features
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_features
    )

    # Adjust the weights of dummy variables according to the number of unique values in the original categorical column
    for encoded_column in df_encoded.columns:
        for original_column in unique_values_dict.keys():
            if original_column + '_' in encoded_column:
                # Get the number of unique values in the original categorical column from the dictionary
                num_unique_values = unique_values_dict[original_column]

                # Adjust the weight of the dummy variable
                df_encoded[encoded_column] = df_encoded[encoded_column] / \
                    np.sqrt(num_unique_values - 1)

                if original_column in categorical_features_to_overweight:
                    df_encoded[encoded_column] = df_encoded[encoded_column] * \
                        categorical_features_overweight_factor

    return df_encoded


def numerical_scaling(df: pd.DataFrame, categorical_features: list, numerical_features: list, numerical_features_to_overweight: list, numerical_features_overweight_factor: int) -> pd.DataFrame:
    """numerical scaling for the bike dataframe
    minmax scaler and apply overweight
    Args:
        df (pandas.DataFrame): dataframe of bikes to feature engineer
        categorical_features (list): list of categorical features
        numerical_features (list): list of numerical features
        numerical_features_to_overweight (list): list of numerical features to overweight
        numerical_features_overweight_factor (int): factor to overweight the numerical features by
    Returns:
        df (pandas.DataFrame): dataframe of bikes feature engineered
    """
    # scale the features
    df[numerical_features] = MinMaxScaler().fit_transform(df[numerical_features])
    # reweight the numerical features according to ratio to categorical features
    df[numerical_features] = df[numerical_features] * \
        len(categorical_features)/len(numerical_features)
    # overweight certain numerical features
    df[numerical_features_to_overweight] = df[numerical_features_to_overweight] * \
        numerical_features_overweight_factor
    return df


def get_similarity_matrix_memory(df: pd.DataFrame, df_feature_engineered: pd.DataFrame, metric: str, working_memory: int = 4001) -> pd.DataFrame:
    """
    get the similarity matrix for the dataframe
    only chunck size optimal, with 4gb not good results, we might need to randomly shuffle similarity_matrix matrix to avoid clustering in chunks
    Args:
        df (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        df_feature_engineered (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        mertric (str): similarity metric
        working_memory (int)
    Returns:
        similarity_matrix (pd.DataFrame): similarity matrix for the dataframe
    """
    # calculate pairwise distances using pairwise_distances_chunked
    n_samples = df_feature_engineered.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))

    for chunk in pairwise_distances_chunked(df_feature_engineered, df_feature_engineered, metric=metric, working_memory=working_memory):
        for i, row in enumerate(chunk):
            similarity_matrix[i, :] = row

    # convert the pairwise distances to a similarity matrix
    similarity_matrix = pd.DataFrame(
        similarity_matrix, columns=df.index, index=df.index)

    return similarity_matrix


def get_similarity_matrix_cdist(df: pd.DataFrame, df_feature_engineered: pd.DataFrame, metric: str, status_mask: pd.DataFrame) -> pd.DataFrame:
    """
    get the similarity matrix for the dataframe
    cdist, since we only need to recommend available bikes (status_mask) but need to recommend for all bikes
    Args:
        df (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        df_feature_engineered (pandas.DataFrame): feature engineered dataframe of bikes to get cosine similarity matrix for
        metric (str): metric to use for pairwise distances
        status_mask (pandas.DataFrame): mask for the status of the bikes
    Returns:
        similarity_matrix (pandas.DataFrame): similarity matrix

    """
    # get the cosine similarity matrix
    similarity_matrix = pd.DataFrame(
        cdist(df_feature_engineered, df_feature_engineered.loc[status_mask]),
        columns=status_mask,
        index=df.index,
    )

    return similarity_matrix


def get_data(
    main_query: str,
    main_query_dtype: str,
    popularity_query: str,
    config_paths: str = "config/config.ini",
) -> pd.DataFrame:
    """
    Query DB, fillna for motor and dropna
    Args:
        config_paths: path to config file
        main_query: query to get main data
        popularity_query: query to get popularity data

    Returns:
        df: main data
        df_popularity: popularity data
    """

    config = configparser.ConfigParser()
    config.read(config_paths)

    user = config["DATABASE"]["user"]
    host = config["DATABASE"]["host_prod"]
    port = int(config["DATABASE"]["port"])
    dbname = config["DATABASE"]["dbname_prod"]
    password = config["DATABASE"]["password_prod"]

    # Create the connection
    engine = create_engine(
        url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(user, password, host, port, dbname))

    df = pd.read_sql_query(sql=text(main_query), con=engine.connect(
    ), index_col="id", dtype=main_query_dtype)

    # currently this excludes around 20% from the recommendations
    # meaning these bikes are not recommended and
    # if they are recommended for the recommendation is based on popularity

    # replace NAs with median of the motor column
    df.motor.fillna(df.motor.median(), inplace=True)
    # df.family_id.fillna(int(df.family_id.median()), inplace=True)

    df.dropna(inplace=True)

    df_popularity = pd.read_sql(
        sql=text(popularity_query), con=engine.connect(), index_col="id")

    return df, df_popularity


def create_data_model_content(
    main_query,
    main_query_dtype,
    popularity_query,
    categorical_features,
    numerical_features,
    prefilter_features,
    numerical_features_to_overweight: list,
    numerical_features_overweight_factor: float,
    categorical_features_to_overweight: list,
    categorical_features_overweight_factor: float,
    status: list,
    metric: str = "eucledian",
    path: str = "data/",

):
    """
    Create data and save it to disk
    includes getting the data,
    replacing the frame_size_code with a numeric value,
    feature engineering the data,
    and getting the similarity matrix
    Args:
        path: path to save data

    """

    df, df_popularity = get_data(
        main_query, main_query_dtype, popularity_query)

    df = frame_size_code_to_numeric(df)

    df_feature_engineered = feature_engineering(df,
                                                categorical_features,
                                                categorical_features_to_overweight,
                                                categorical_features_overweight_factor,
                                                numerical_features,
                                                numerical_features_to_overweight,
                                                numerical_features_overweight_factor)

    status_mask = get_data_status_mask(df, status)

    similarity_matrix = get_similarity_matrix_cdist(
        df, df_feature_engineered, metric, status_mask)

    # reduce the column dimensionality of the similarity matrix by filtering with the status mask
    # similarity_matrix = similarity_matrix[status_mask]

    df_status_masked = df.loc[status_mask]
    df = df[prefilter_features]
    df_status_masked = df_status_masked[prefilter_features]

    # write df, df_popularity, similarity_matrix to disk

    df.to_pickle(path + "df.pkl")  # where to save it, usually as a .pkl
    df_status_masked.to_pickle(path + "df_status_masked.pkl")
    df_popularity.to_pickle(path + "df_popularity.pkl")
    similarity_matrix.to_pickle(
        path + "similarity_matrix.pkl", compression="tar")


def read_data_content(path: str = "data/"):
    """
    Read data from disk
    Args:
        path: path to save data

    Returns:
        df: main data
        df_status_masked: main data with status mask applied
        df_popularity: popularity data
        similarity_matrix: similarity matrix
    """

    df = pd.read_pickle(path + "df.pkl")
    df_status_masked = pd.read_pickle(path + "df_status_masked.pkl")
    df_popularity = pd.read_pickle(path + "df_popularity.pkl")

    similarity_matrix = pd.read_pickle(
        path + "similarity_matrix.pkl", compression="tar")

    return df, df_status_masked, df_popularity,  similarity_matrix


class DataStoreContent:
    def __init__(self):
        self.df = None
        self.df_status_masked = None
        self.df_popularity = None
        self.similarity_matrix = None

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
                self.df, self.df_status_masked, self.df_popularity, self.similarity_matrix = read_data_content()

                logger.info("Data read",
                            extra={
                                "df_shape": self.df.shape,
                                "df_status_masked_shape": self.df_status_masked.shape,
                                "df_popularity_shape": self.df_popularity.shape,
                                "similarity_matrix_shape": self.similarity_matrix.shape,
                            })

            except Exception as error:
                logger.error("Data could not be read: " + str(error))
            sleep(period)

