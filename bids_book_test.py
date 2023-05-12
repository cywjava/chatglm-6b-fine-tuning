# coding=UTF-8
from gpt_db_tools.dbutil import DBUtil
from bids_book_industry_server import find_response_items

dbutil = DBUtil('192.168.10.30', '12277', 'userapp', '1Qaz2Wsx', 'ai')
dbutil.print_db_info()


def test_bik():
    question_list = dbutil.query_question_by_sqid("bids_book_20230511212152", "chenyw8", "8")
    answer1 = question_list[1]
    print(answer1)
    content_items = find_response_items(answer1)
    print(content_items)


test_bik()
