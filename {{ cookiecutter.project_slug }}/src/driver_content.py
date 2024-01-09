import pandas as pd

# features for which to prefilter for the specific recommendations.
# currently only once feature is supported, check .all or .any for supporting multiple features
# however not sure if that makes sense, maybe .any for family and family_model
# needs some thought

#prefilter_features = ["family_model_id", "family_id", "brand_id"]
prefilter_features = ["family_id"]

# for the content based recommendation we disregard prefilter_features and use generic features that represent the qualities of the bike
# categorical_features and numerical_features features to consider in the generic recommendations

categorical_features = [
    "motor",
    "bike_component_id",
    "bike_category_id",
    "bike_type_id",
    "brake_type_code",
    "frame_material_code",
    "shifting_code",
    "color",
]
numerical_features = ["price", "frame_size_code", "year"]

# features to overweight
numerical_features_to_overweight = ["price", "frame_size_code"]
numerical_features_overweight_factor = 4
categorical_features_to_overweight = ["bike_component_id", "bike_category_id", "bike_type_id"]
categorical_features_overweight_factor = 2

# main query, needs to include at least the id and the features defined above
main_query = """SELECT id,

                       status,
                       -- categorizing
                       bike_type_id,
                       bike_category_id,
                       motor,

                       -- cetegorizing fuzzy

                       frame_size_code,


                       -- very important
                       price,
                       brake_type_code,
                       frame_material_code,
                       shifting_code,


                       -- important

                       year,

                       bike_component_id,

                       -- find similarity between hex codes
                       color,

                       -- quite specific
                       family_id

                FROM bikes


                -- for non active bikes we set a one year cap for updated_at
                WHERE status = 'active' or status != 'new' and TIMESTAMPDIFF(MONTH, updated_at, NOW()) < 4


             """

main_query_dtype = {
    "id": pd.Int64Dtype(),
    "status": pd.StringDtype(),
    "bike_type_id": pd.Int64Dtype(),
    "bike_category_id": pd.Int64Dtype(),
    "motor": pd.Int64Dtype(),
    "price": pd.Float64Dtype(),
    "year": pd.Int64Dtype(),
    "bike_component_id": pd.Int64Dtype(),
     "family_id": pd.Int64Dtype(),
     # frame_size_code as string, we convert it in frame_size_code_to_numeric
    "frame_size_code": pd.StringDtype(),
}


popularity_query = """SELECT id,
                       frame_size_code,
                       price,
                       family_id


                FROM bikes

                -- not sure if this is correct
                -- we might also need recommendations for bikes that are not active
                WHERE status = 'active'



                ORDER BY count_of_visits DESC

             """


