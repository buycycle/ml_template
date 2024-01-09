
user_id = 'distinct_id'
bike_id = 'bike.id'


item_features = [
    # 'bike_type_id',
    'bike.family_id',
    'bike.frame_size',
    'bike_price',
]

user_features = [
]

features = [user_id] + [bike_id] + user_features + item_features




implicit_feedback = {'Bike_view': 1,
                     'Choose_service': 5,
                     'Choose_shipping_method': 5,
                     'Save_filter': 5,
                     'add_discount': 10,
                     'add_payment_info': 20,
                     'add_shipping_info': 20,
                     'add_to_compare': 3,
                     'add_to_favorite': 10,
                     'ask_question': 10,
                     'begin_checkout': 12,
                     'choose_condition': 0,
                     'click_filter_no_result': 0,
                     'close_login_page': 0,
                     'comment_show_original': 3,
                     'counter_offer': 10,
                     'delete_from_favourites': -2,
                     'home_page_open': 0,
                     'login': 5,
                     'open_login_page': 2,
                     'purchase': 50,
                     'receive_block_comment_pop_up': 0,
                     'recom_bike_view': 3,
                     'register': 10,
                     'remove_bike_from_compare': -1,
                     'request_leasing': 20,
                     'sales_ad_created': 0,
                     'search': 0,
                     'search_without_result': 0,
                     'sell_click_family': 0,
                     'sell_click_template': 0,
                     'sell_condition': 0,
                     'sell_details': 0,
                     'sell_search': 0,
                     'sell_templates': 0,
                     'sell_toggle_components_changed': 0,
                     'sellpage_open': 0,
                     'share_bike': 10,
                     'shop_view': 2,
                     'toggle_language': 2,
                     }


