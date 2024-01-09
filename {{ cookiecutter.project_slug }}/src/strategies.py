

import pandas as pd


from abc import ABC, abstractmethod

from src.content import get_top_n_recommendations
from src.content import get_top_n_recommendations_prefiltered
from src.content import get_top_n_popularity_prefiltered
from src.content import get_top_n_recommendations_mix

from src.collaborative import get_top_n_collaborative, read_data_model


class RecommendationStrategy(ABC):

    @abstractmethod
    def get_recommendations(self, bike_id: int, n: int) -> list:
        pass


class PopularityStrategy(RecommendationStrategy):

    def __init__(self, df_popularity: pd.DataFrame, family_id: int, price: int, frame_size_code: str):
        self.df_popularity = df_popularity
        self.family_id = family_id
        self.price = price
        self.frame_size_code = frame_size_code

    def get_recommendations(self, bike_id: int, n: int) -> list:
        return get_top_n_popularity_prefiltered(self.df_popularity, self.family_id, self.price, self.frame_size_code, n)


class GenericStrategy(RecommendationStrategy):

    def __init__(self, similarity_matrix: pd.DataFrame):
        self.similarity_matrix = similarity_matrix

    def get_recommendations(self, bike_id: int, n: int) -> list:
        return get_top_n_recommendations(self.similarity_matrix, bike_id, n)


class PrefilterStrategy(RecommendationStrategy):

    def __init__(self, similarity_matrix: pd.DataFrame, df: pd.DataFrame, df_status_masked: pd.DataFrame, prefilter_features: list):
        self.similarity_matrix = similarity_matrix
        self.df = df
        self.df_status_masked = df_status_masked
        self.prefilter_features = prefilter_features

    def get_recommendations(self, bike_id: int, n: int) -> list:
        return get_top_n_recommendations_prefiltered(self.similarity_matrix, self.df, self.df_status_masked, bike_id, self.prefilter_features, n)


class MixedStrategy(RecommendationStrategy):

    def __init__(self, df: pd.DataFrame, df_status_masked: pd.DataFrame, df_popularity: pd.DataFrame, similarity_matrix: pd.DataFrame, prefilter_features: list, logger):
        self.df = df
        self.df_status_masked = df_status_masked
        self.df_popularity = df_popularity
        self.similarity_matrix = similarity_matrix
        self.prefilter_features = prefilter_features
        self.logger = logger

    def get_recommendations(self, bike_id: int, family_id: int, price: int, frame_size_code: str, n: int) -> tuple:
        return get_top_n_recommendations_mix(
            bike_id,
            family_id,
            price,
            frame_size_code,
            self.df,
            self.df_status_masked,
            self.df_popularity,
            self.similarity_matrix,
            self.prefilter_features,
            self.logger,
            n,
            ratio=0.5,
            interveave_prefilter_general=False,
        )


class CollaborativeStrategy(RecommendationStrategy):

    def __init__(self, path: str, n: int, logger):
        self.model, self.dataset = read_data_model(path)
        self.n = n
        self.logger = logger

    def get_recommendations(self, user_id: str, n: int) -> tuple:
        return get_top_n_collaborative(
            self.model,
            user_id,
            self.n,
            self.dataset,
            self.logger,
        )
