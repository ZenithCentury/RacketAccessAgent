import random
import mysql.connector
# from core.core_function import RacketRecommender

def racket_load():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='dongge233',
        database='racket'
    )
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM racket_property "
    cursor.execute(query)
    results = cursor.fetchall()
    return results

def Ulike(num):
    items = racket_load()
    random_values = random.sample(items, num)
    print(random_values)
    return random_values

def racket_name2all(name_list):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='dongge233',
        database='racket'
    )
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM racket_property WHERE racket_name IN (%s) ORDER BY FIELD(racket_name, %s)" % (','.join(['%s']*len(name_list)) , ','.join(['%s']*len(name_list)))
    cursor.execute(query, name_list * 2)
    results = cursor.fetchall()

    return results

# def Recommend4Me(my_items):
#     #提取出my_items的球拍名和对应的评分
#
#
#     print('正在为你初始化推荐算法')
#     racket_recommender = RacketRecommender()
#     print('正在获取新数据')
#     new_user = pd.DataFrame({
#         'user_id': 'new_user',
#         'item_id': ["锋影800NEW", "THRUSTER RYUGA Ⅱ PRO", "THRUSTER F 隼 SE", "ASTROX 77 PRO 深橙色",
#                     "ASTROX 88S PRO 银/黑", "雷霆60", "锋影900MAX日", "AURASPEED 100X", "THRUSTER RYUGA Ⅱ", "战戟6000",
#                     "战戟9000", "AURASPEED 90K Ⅱ", "AURASPEED 90K METALLIC", "战戟8000", "THRUSTER TTY",
#                     "ASTROX 100ZZ 古红色", "THRUSTER F 隼 黑金", "TECTONIC 9"],
#         'rating': [5, 5, 5, 5, 5, 3, 5, 4, 3, 5, 5, 4, 5, 5, 5, 5, 5, 4]
#
#     })

