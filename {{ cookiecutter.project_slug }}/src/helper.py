from typing import Any
import pandas as pd

def interveave(list1: list, list2: list) -> list:
    """ interveave two lists"""

    return [item for x in zip(list1, list2) for item in x] + (list2[len(list1):] if len(list2) > len(list1) else list1[len(list2):])


def get_field_value(dataframe: pd.DataFrame, field: str, default_value: Any, dtype: type = int) -> Any:
    """
    Function to get the field value from a DataFrame or return a default value if the field does not exist
    or has a null value.

    Parameters:
    dataframe (pd.DataFrame): DataFrame to extract the field value from.
    field (str): Field/column name to extract from the dataframe.
    default_value (Any): Default value to return if the field does not exist or its value is null.
    dtype (type): Datatype to convert the field value to. Default is int.

    Returns:
    Any: The field value from the DataFrame converted to the specified datatype.
    If the field does not exist or its value is null, the default value is returned.
    """
    if field not in dataframe or pd.isnull(dataframe[field].iloc[0]):
        return dtype(default_value)
    else:
        return dtype(dataframe[field].iloc[0])
