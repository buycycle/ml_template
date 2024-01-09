
import sys
from data_content import create_data_model_content

from driver_content import main_query, main_query_dtype, popularity_query, categorical_features, numerical_features, prefilter_features, numerical_features_to_overweight, numerical_features_overweight_factor, categorical_features_to_overweight, categorical_features_overweight_factor

from driver_collaborative import user_id, bike_id, features, item_features, user_features, implicit_feedback
from data_collaborative import s3_client, s3_credentials, get_last_date_S3, get_last_date_local, get_multi_day_data, write_data, combine_data
from collaborative import create_data_model_collaborative, update_model

# if there is a command line argument, use it as path, else use './data/'
path = sys.argv[1] if len(sys.argv) > 1 else "./data/"

print('create_data_model_content')
create_data_model_content(
    main_query=main_query,
    main_query_dtype=main_query_dtype,
    popularity_query=popularity_query,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    prefilter_features=prefilter_features,
    numerical_features_to_overweight=numerical_features_to_overweight,
    numerical_features_overweight_factor=numerical_features_overweight_factor,
    categorical_features_to_overweight=categorical_features_to_overweight,
    categorical_features_overweight_factor=categorical_features_overweight_factor,
    status=["active"],
    metric="euclidean",
    path=path,
)

print('create_data_model_collaborative')
create_data_model_collaborative(s3_client=s3_client,
                                s3_credentials=s3_credentials,
                                get_last_date_S3=get_last_date_S3,
                                get_last_date_local=get_last_date_local,
                                get_multi_day_data=get_multi_day_data,
                                implicit_feedback=implicit_feedback,
                                features=features,
                                user_features=user_features,
                                item_features=item_features,
                                user_id=user_id,
                                bike_id=bike_id,
                                combine_data=combine_data,
                                write_data=write_data,
                                update_model=update_model,
                                path=path,
                                limit=2,
                                fraction=0.1)
