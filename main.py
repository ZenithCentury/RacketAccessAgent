import re
import sys

import mysql.connector
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import *
from racket_load import Ulike


#主界面，涵盖通往其他界面的按钮  和一总搜索系统，配合往 我的装备 加入装备
class MAINWINDOW(QMainWindow):

    def __init__(self):
        super().__init__()

        #预参数
        self.chosen_item = None
        self.my_items = []
        self.my_rate = []
        self.show_page= None
        self.recommend_list = None

        #硬件加速
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        # 主页面布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        #初始化主窗体
        self.setAttribute(Qt.WA_TranslucentBackground)  # 窗体背景透明
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # 窗口置顶，无边框
        self.setWindowTitle("董同一")
        self.resize(2300, 1400)  # 设置窗体大小
        self.setWindowIcon(QIcon('logo.png'))
        self.center_on_screen()
        #背景路径
        self.background = QPixmap("./UI/background.png")  # 预加载
        self.backpic = QPixmap("./UI/backpic.png")
        # central = HOMEPAGE()
        # self.setCentralWidget(central)

        self.init_ui()

        # 主页面搜索框
        # 防抖  500ms内只发一次   快速键入内容则重执
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.do_search)

    # 窗口中心化
    def center_on_screen(self):
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        # 将窗口的中心点设置为屏幕的中心点
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    # 设置背景
    def paintEvent(self, a0: QtGui.QPaintEvent, **kwargs) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # 平滑缩放
        painter.drawPixmap(self.rect(), self.backpic)
        painter.drawPixmap(self.rect(), self.background)

    #清空搜索框
    def dynamic_clear_search(self):
        #安全删除小组件
        if self.dynamic_content_search.layout():
            while self.dynamic_content_search.layout().count():
                item = self.dynamic_content_search.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def init_ui(self):

        # 创建独立动态内容区域-其他
        self.dynamic_content = QWidget(self)
        self.dynamic_content.setParent(None)
        self.dynamic_content.setParent(self)
        self.dynamic_content.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 10px;")
        self.dynamic_content.setGeometry(QRect(504, 224,1685, 1088))
        # 关键修改：设置布局
        self.dynamic_layout = QVBoxLayout(self.dynamic_content)  # 直接作为主布局
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        self.show_random()

        # 创建独立动态内容区域-搜索栏
        self.dynamic_content_search = QWidget(self)
        self.dynamic_layout_search = QVBoxLayout()
        self.dynamic_content_search.setVisible(False)  # 初始隐藏
        self.dynamic_content_search.setParent(self)
        self.dynamic_content_search.setStyleSheet("background-color: rgba(255, 255, 255, 0.7); border-radius: 10px; padding: 10px;")
        self.dynamic_content_search.setGeometry(QRect(619, 182, 744, 248))
        self.dynamic_content_search.setLayout(self.dynamic_layout_search)

        # 清空动态内容
        # 清空搜索栏部分
        # self.dynamic_clear()

        # 应用主题
        # 按钮主题
        button_style = """
                                    QPushButton {
                                        background-color: rgba(255, 255, 255, 0.0);
                                        border: none;
                                        border-radius: 0px;
                                        color: white;
                                        padding: 15px;
                                        font-size: 16px;
                                    }
        
                                    QPushButton:pressed {
                                        background-color: rgba(255,255,255, 1.0);
                                    }
                                    QPushButton:hover {
                                        background-color: rgba(0, 0, 0, 0.1);
                                    }
                                """

        #布局
        layout = QVBoxLayout()

        # 按钮
        self.button1 = QPushButton(self)
        icon1 = QIcon('./UI/button1.png')
        self.button1.setIcon(icon1)
        self.button1.setIconSize(QSize(391, 103))
        # 设置按钮时关闭自动重绘
        self.button1.setUpdatesEnabled(False)
        self.button1.setGeometry(QtCore.QRect(30, 289, 391, 102))
        self.button1.setUpdatesEnabled(True)  # 重新启用
        self.button1.setStyleSheet(button_style)
        self.button1.clicked.connect(self.show_random)
        layout.addWidget(self.button1)

        self.button2 = QPushButton(self)
        icon1 = QIcon('./UI/button2.png')
        self.button2.setIcon(icon1)
        self.button2.setIconSize(QSize(391, 103))
        # 设置按钮时关闭自动重绘
        self.button2.setUpdatesEnabled(False)
        self.button2.setGeometry(QtCore.QRect(30, 391, 391, 102))
        self.button2.setUpdatesEnabled(True)  # 重新启用
        self.button2.setStyleSheet(button_style)
        self.button2.clicked.connect(self.my_item_show_items)
        layout.addWidget(self.button2)

        self.button3 = QPushButton(self)
        icon1 = QIcon('./UI/button3.png')
        self.button3.setIcon(icon1)
        self.button3.setIconSize(QSize(391, 103))
        # 设置按钮时关闭自动重绘
        self.button3.setUpdatesEnabled(False)
        self.button3.setGeometry(QtCore.QRect(30, 494, 391, 102))
        self.button3.setUpdatesEnabled(True)  # 重新启用
        self.button3.setStyleSheet(button_style)
        # self.button3.clicked.connect()
        layout.addWidget(self.button3)

        self.button4 = QPushButton(self)
        icon1 = QIcon('./UI/button4.png')
        self.button4.setIcon(icon1)
        self.button4.setIconSize(QSize(391, 103))
        # 设置按钮时关闭自动重绘
        self.button4.setUpdatesEnabled(False)
        self.button4.setGeometry(QtCore.QRect(30, 596, 391, 102))
        self.button4.setUpdatesEnabled(True)  # 重新启用
        self.button4.setStyleSheet(button_style)
        self.button4.clicked.connect(self.do_recommend)
        layout.addWidget(self.button4)

        self.button5 = QPushButton(self)
        # 设置按钮时关闭自动重绘
        self.button5.setUpdatesEnabled(False)
        self.button5.setGeometry(QtCore.QRect(1270, 96, 96, 86))
        self.button5.setUpdatesEnabled(True)  # 重新启用
        self.button5.setStyleSheet(button_style)
        # self.button5.clicked.connect()
        layout.addWidget(self.button5)

        self.search = QLineEdit(self)
        self.search.resize(648,84)
        self.search.setGeometry(QtCore.QRect(622, 96, 648, 84))
        self.search.setPlaceholderText('战戟 8000')
        self.search.setStyleSheet("""
                                                QLineEdit {
                                                    border: none;
                                                    background-color:  rgba(255, 255, 255, 0.0);
                                                    font-size: 37px;
                                                    color: #668572; 
                                                    padding: 20px;
                                                }
                                            """)
        layout.addWidget(self.search)
        #如果被交互 则弹出搜索框
        self.search.textChanged.connect(self.on_search)

    #_______________________________________按钮交互_____________________________________________

    def show_random(self):
        item = Ulike(10)
        self.dynamic_show_items(item)

    def do_recommend(self):

        print("先选择查看或者推荐")
        self.dynamic_clear()
        my_item_list = self.my_items

        self.recommend_page = SHOWPAGE(self.recommend_list, self.dynamic_content, self)  # 关键修改：传递主窗口引用
        print(my_item_list)
        if my_item_list == []:
            print(2)
            self.recommend_page.if_empty()
            self.dynamic_layout.addWidget(self.recommend_page)
        else:
            if self.recommend_list == []:
                self.recommend_page.no_recommendation()
            else:
                self.recommend_page.apply_recommend()
            
        self.dynamic_layout.addWidget(self.recommend_page)
        self.recommend_page.recommend_signal.connect(self.recommend)

    #_______________________________________搜索框交互_______________________________________________
    def on_search(self):
        self.search_timer.stop()
        self.search_timer.start(500)

    #进行对录入关键词进行搜索
    def do_search(self):
        keyword = self.search.text().strip()#删除空格
        test = bool(re.search(r'[^\w\s\u4e00-\u9fff]+', keyword))
        if not keyword or test:
            self.dynamic_content_search.setVisible(False)
            print(3)
            return
        self.dynamic_content_search.setVisible(True)
        self.search_result(keyword)

    def search_result(self,keyword):
        try:
            # 清空之前的结果
            # self.clear_dynamic_content()

            # 获取搜索结果
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='dongge233',
                database='racket'
            )
            cursor = conn.cursor(dictionary=True)

            query = "SELECT * FROM racket_property WHERE racket_name LIKE %s"
            cursor.execute(query, ('%' + keyword + '%',))
            self.results = cursor.fetchall()
            print(self.results)


            if not self.results:
                self.dynamic_clear_search()
                no_result = QLabel("没有找到匹配的商品")
                no_result.setStyleSheet("font-size: 40px;")
                self.dynamic_layout_search.addWidget(no_result, 0)
            else:
                print(5)
                self.dynamic_clear_search()
                # 显示结果列表
                search_results_list = QListWidget()
                search_results_list.setAlternatingRowColors(True)  # 交替行颜色
                search_results_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏垂直滚动条
                search_results_list.setStyleSheet("""
                                                                QListWidget {
                                                                    font-size: 24px;
                                                                }
                                                                QListWidget::item {
                                                                    padding: 5px;
                                                                }
                                                            """)
                for item in self.results:
                    search_results_list.addItem(f"{item['racket_name']} - {item.get('racket_brand', '未知')}")
                    search_results_list.item(search_results_list.count() - 1).setData(Qt.UserRole, item)
                self.dynamic_layout_search.addWidget(search_results_list)
                search_results_list.itemDoubleClicked.connect(self.go_search)


        except mysql.connector.Error as err:
            error_label = QLabel(f"数据库错误: {err}")
            self.dynamic_layout_search.addWidget(error_label)
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    def go_search(self, item):
        racket_data = item.data(Qt.UserRole)
        self.chosen_item = racket_data
        self.open_detail_page()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.dynamic_content_search.isVisible():
            self.dynamic_content_search.setVisible(False)
        super().keyPressEvent(event)

    #---------------------------以下函数用以在主窗口中的活动区打开其他窗口---------------------------------

    #清空子页面动态区域
    def dynamic_clear(self):
        # 安全删除小组件
        if self.dynamic_content.layout():
            while self.dynamic_content.layout().count():
                item = self.dynamic_content.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    #动态区域商品展示
    def dynamic_show_items(self,show_items_list):
        self.dynamic_clear()
        self.show_page = 1
        print('主窗口收到')
        item_page = SHOWPAGE(show_items_list, self.dynamic_content,self)  # 关键修改：传递主窗口引用
        self.dynamic_layout.addWidget(item_page)

    #接受信号(目标  打开页面)
    def open_detail_signal_connect(self,item):
        self.chosen_item = item #选定item的全信息存入全局item （一次只娶一个）
        print('已录入')
        self.open_detail_page()

    #转到商品本身
    def open_detail_page(self):
        print('已转达')
        self.dynamic_clear()
        if not self.chosen_item in self.my_items:
            print('打开一般进入的详情界面')
            detail_page = ITEMDETAILPAGE(self.chosen_item, self.dynamic_content)
            detail_page.open_detail_signal2.connect(self.open_detail_signal_add)
            detail_page.open_detail_signal3.connect(self.open_detail_signal_empty)
            detail_page.open_detail_signal0.connect(self.add_first_rate)
            self.dynamic_layout.addWidget(detail_page)

        else :
            print('打开由我的装备界面进入的详情界面,同时给到得分')
            num = self.my_items.index(self.chosen_item)
            detail_page = MYITEMDETAILPAGE(self.chosen_item, self.dynamic_content, self.my_rate[num])
            detail_page.open_detail_signal4.connect(self.open_detail_signal_rate)
            detail_page.open_detail_signal5.connect(self.open_detail_signal_empty)
            self.dynamic_layout.addWidget(detail_page)

        # 接受信号（目标  评分本装备）
    #第一次
    def add_first_rate(self,rate):
        print('已转入加分')
        self.my_rate.append(rate)
        print(f"目前我的 {self.my_rate}")
        self.my_item_show_items()

    #修改
    def open_detail_signal_rate(self, new_rate):
        print('新评分')
        self.my_rate[self.my_items.index(self.chosen_item)] = new_rate
        print(f'已更新评分{self.my_rate}')


    #接受信号（目标  加入我的装备）
    def open_detail_signal_add(self, item):
        if item in self.my_items:
            print('已有')
            return
        self.my_items.append(item)
        print(f"目前我的 {self.my_items}")

    #接受信号（目标 删除本装备）
    def open_detail_signal_empty(self,item):
        print('清空收到')
        if item in self.my_items :
            num = self.my_items.index(item)
            print(num)
            del self.my_items[num]
            del self.my_rate[num]
        self.my_item_show_items()

    def my_item_show_items(self):
        show_items_list = self.my_items
        self.dynamic_clear()
        self.show_page = 2
        self.my_item_page = SHOWPAGE(show_items_list, self.dynamic_content, self)  # 关键修改：传递主窗口引用
        if show_items_list == []:
            self.my_item_page.if_empty()
        else:
            self.my_item_page.not_empty()
        print('我的装备收到')
        self.dynamic_layout.addWidget(self.my_item_page)


#展品    对于每个会出现在表单中的展品 统一以如下方式进行展示
class ITEMCARD(QFrame):
    open_detail_signal = pyqtSignal(dict)  # 定义信号，传递商品数据

    def __init__(self, item, parent=None):
        super().__init__(parent)            #从父类获取商品单
        self.item = item
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)
        self.setStyleSheet("""
                                        ITEMCARD {
                                            background: rgba(255, 255, 255, 0.0);
                                            border-radius: 5px;
                                            padding: 10px;
                                            margin-bottom: 10px;
                                        }
                                        QLabel#title {
                                            font-size: 30px;
                                            font-weight: bold;
                                        }
                                        QLabel#price {
                                            color: red;
                                            font-size: 20px;
                                        }
                                        QLabel#score {
                                            color: gray;
                                            font-size: 20px;
                                        }
                                        ITEMCARD:hover{
                                            background-color: rgba(0, 0, 0, 0.1);
                                        }
                                        
                                        
                                    """)

        self.setLayout(QVBoxLayout())

        # 商品标题
        title = QLabel(item.get('racket_name', '未命名'))
        title.setObjectName("title")
        self.layout().addWidget(title)

        # 商品价格
        price = QLabel(f"价格: ¥{item.get('racket_price_range', '0.00')}")
        price.setObjectName("price")
        self.layout().addWidget(price)

        # 中羽评分
        score = QLabel(f"中羽热度: {item.get('racket_score', 0)}")
        score.setObjectName("score")
        self.layout().addWidget(score)

    #重写点击时间
    def mousePressEvent(self, event):
        self.open_detail_signal.emit(self.item)#发送信号，数据为 item的全部数据（名称价格等）

#子界面1  包含商品展柜：展柜大小是动态的                   该子界面需要从外部拿到展览单
class SHOWPAGE(QWidget):
    recommend_signal = pyqtSignal(bool)

    def __init__(self, items=None, parent=None, hub=None): #items是展览单  hub是为了直接连接到MAINWINDOW
        super().__init__(parent)
        self.items = items
        self.hub = hub
        self.setLayout(QVBoxLayout())
        self.setAttribute(Qt.WA_StyledBackground)  # 关键属性
        self.setObjectName("showPage")  # 设置唯一ID
        self.setStyleSheet("""
                                        #showPage {
                                            background: rgba(255, 255, 255, 0.5);
                                            border: 1px solid #ccc;
                                        }
                                        
                                        /* 滚动区域样式（不直接控制滚动条） */
                                        #showPage QScrollArea {
                                        background: transparent;
                                        border: none;
                                        }
                        
                                        /* 垂直滚动条全局样式 */
                                        #showPage QScrollBar:vertical {
                                        width: 12px;
                                        background: rgba(200, 200, 200, 0.2);
                                        margin: 2px;
                                        }
                        
                                        /* 垂直滚动条手柄 */
                                        #showPage QScrollBar::handle:vertical {
                                        background: #a0a0a0;
                                        min-height: 30px;
                                        border-radius: 6px;
                                        }
                        
                                        /* 垂直滚动条手柄悬停状态 */
                                        #showPage QScrollBar::handle:vertical:hover {
                                        background: #808080;
                                        }
                        
                                        /* 垂直滚动条上下按钮 */
                                        #showPage QScrollBar::add-line:vertical,
                                        #showPage QScrollBar::sub-line:vertical {
                                        height: 0px;  /* 隐藏默认按钮 */
                                        }
                                        
                                        #showPage QPushButton {
                                        background-color: rgba(255, 255, 255, 0.0);
                                        border: none;
                                        border-radius: 0px;
                                        color: white;
                                        padding: 15px;
                                        font-size: 16px;                                            
                                        }
                    
                                        QPushButton:pressed {
                                        background-color: rgba(255,255,255, 1.0);
                                        }
                                        
                                        QPushButton:hover {
                                        background-color: rgba(0, 0, 0, 0.1);
                                        }
    
                                    """)

        #创建表单区
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setAlignment(Qt.AlignTop)

        # 创建滚动区域
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        print(9)
        if self.items is not None:
            # 添加商品卡片
            for item in self.items:
                print(10)
                card = ITEMCARD(item)
                # 接受到信号挂载的信息
                card.open_detail_signal.connect(
                    self.hub.open_detail_signal_connect)  # 如果点击卡片  则发送信号  该信号将连接到父类 即召唤主窗口中show_item_detail函数 并输入参数item的全部信息

                self.container_layout.addWidget(card)

            # 将全部表单加入到滚动区
        self.scroll.setWidget(self.container)
        self.layout().addWidget(self.scroll)

    def if_empty(self):
        empty = QLineEdit()
        empty.setPlaceholderText("您的装备库为空，您可通过搜索或主页进入到商品界面以添加装备")
        empty.setStyleSheet("font-size: 45px; color: rgba(55, 75, 85, 1.0);")
        empty.setReadOnly(True)
        empty.setAlignment(Qt.AlignTop)
        self.layout().insertWidget(0,empty)

    def not_empty(self):
        nempty = QLineEdit()
        nempty.setPlaceholderText("通过对装备打分，您可以获得效果更好的推荐")
        nempty.setStyleSheet("font-size: 45px; color: 097A7A;")
        nempty.setReadOnly(True)
        nempty.setAlignment(Qt.AlignTop)
        nempty.setGeometry(QRect(504, 1212, 1413, 100))
        self.layout().addWidget(nempty)

    def no_recommendation(self):
        empty = QLineEdit()
        empty.setPlaceholderText("您的推荐列表为空，请点击下方按钮进行推荐或重新推荐")
        empty.setStyleSheet("font-size: 45px; color: rgba(55, 75, 85, 1.0);")
        empty.setReadOnly(True)
        empty.setAlignment(Qt.AlignTop)
        self.layout().insertWidget(0, empty)

    def apply_recommend(self):
        button_recommend = QPushButton('点击为您推荐')
        button_recommend.setGeometry(QRect(1917, 1168, 223, 94))
        button_style = """
                                    QPushButton {
                                        background-color: rgb(0, 203, 154, 0.6);
                                        border: none;
                                        border-radius: 0px;
                                        color: purple;
                                        padding: 15px;
                                        font-size: 40px;
                                    }

                                    QPushButton:pressed {
                                        background-color: rgba(255,255,255, 1.0);
                                    }
                                    QPushButton:hover {
                                        background-color: rgba(0, 0, 0, 0.1);
                                    }
                                """
        button_recommend.setStyleSheet(button_style)
        button_recommend.clicked.connect(self.send_recommend_signal)
        self.layout().addWidget(button_recommend)

    def send_recommend_signal(self):
        self.recommend_signal.emit(True)

#咨询弹窗
class ChatDialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog)
        self.setWindowModality(Qt.ApplicationModal)
        self.setGeometry(QRect(646, 299, 1350, 1064))
        self.background = QPixmap("./UI/backpic.png")  # 预加载
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("background-color: rgba(255, 255, 255, 0.9);")

        self.setLayout(QGridLayout())



class ConversationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowModality(Qt.ApplicationModal)  # 阻塞整个程序
        self.setGeometry(QRect(506, 224, 1330, 1400))
        self.setLayout(QGridLayout())
        self.init_ui()


    def init_ui(self):
        self.question_area = QTextEdit()
        self.question_area.setReadOnly(True)
        self.question_area.setPlaceholderText('等待智能体回复')
        self.question_area.setStyleSheet("background-color: rgba(0,0,0,0.2);")
        self.question_area.setGeometry(QRect(558, 274, 1226, 587))

        self.layout.addWidget(self.question_area, 0, 0, 1, 7)









#商品独立介绍界面
class ITEMDETAILPAGE(QWidget):
    #发送信号标志接受本商品信息到我的装备
    open_detail_signal2 = QtCore.pyqtSignal(dict)
    #发送信号表示删除本装备在我的装备中的信息
    open_detail_signal3 = QtCore.pyqtSignal(dict)
    #发送信号给与初始评分
    open_detail_signal0 = QtCore.pyqtSignal(int)

    def __init__(self, item = None, parent=None):
        super().__init__(parent)
        self.item_data = item
        self.init_ui()
        self.this_rate = None

    def init_ui(self):
        self.setWindowTitle(self.item_data.get('racket_name', '商品详情'))
        self.resize(800, 600)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 0.6);")
        self.setLayout(QGridLayout())

        # 商品信息
        title_label = QLabel(self.item_data.get('racket_name', '未命名'))
        title_label.setStyleSheet("font-size: 40px; font-weight: bold; color: black;")
        self.layout().addWidget(title_label, 0, 0, 1, 5)

        brand_label = QLabel(f"品牌: {self.item_data.get('racket_brand', '董哥')}")
        brand_label.setStyleSheet("font-size: 32px; color: black;")
        self.layout().addWidget(brand_label, 2, 0, 1, 5)

        price_label = QLabel(f"价格: ￥{self.item_data.get('racket_price_range', 0)}")
        price_label.setStyleSheet("font-size: 32px; color: red;")
        self.layout().addWidget(price_label, 4, 0, 1, 5)

        score_label = QLabel(f"中羽得分（仅供参考）: {self.item_data.get('racket_score', 0)}")
        score_label.setStyleSheet("font-size: 32px; color: green;")
        self.layout().addWidget(score_label, 6, 0, 1, 5)

        weight_label = QLabel(f"拍重 : {self.item_data.get('racket_weight', 0)}")
        weight_label.setStyleSheet("font-size: 32px; color: green;")
        self.layout().addWidget(weight_label, 8, 0, 1, 7)


        more_label = QLabel(f"更多信息:\n 拍杆材质: {self.item_data.get('racket_stick', '暂无描述')}"
                              f"\n 拍框材质: {self.item_data.get('racket_frame', '暂无描述')} "
                              f"\n 顶磅: {self.item_data.get('racket_max_pounds', '暂无描述')}"
                              f"\n 杆韧: {self.item_data.get('racket_toughness', '暂无描述')}"
                              f"\n 杆粗:  {self.item_data.get('racket_thickness', '暂无描述')}"
                              f"\n 杆长:  {self.item_data.get('racket_length', '暂无描述')}"
                              f"\n 平衡:  {self.item_data.get('racket_balancepoint', '暂无描述')}")
        more_label.setWordWrap(True)  # 自动换行
        more_label.setStyleSheet("font-size: 32px;")
        more_label.setAlignment(Qt.AlignLeft)
        self.layout().addWidget(more_label, 9, 0, 3, 6)

        #创建加入按钮
        button_add = QPushButton('加入装备')
        button_add_style = '''
                                        QPushButton {
                                            background-color: rgba(20, 20, 20, 0.5);
                                            border: none;
                                            border-radius: 0px;
                                            color: white;
                                            padding: 15px;
                                            font-size: 36px;
                                        }

                                        QPushButton:pressed {
                                            background-color: rgba(255,255,255, 1.0);
                                        }
                                        QPushButton:hover {
                                            background-color: rgba(0, 0, 0, 0.1);
                                        }
        
                                    '''
        button_add.setStyleSheet(button_add_style)
        button_add.setFixedSize(190, 94)
        self.layout().addWidget(button_add, 9, 6, 1, 1)
        button_add.clicked.connect(self.send_item)

        #创建删除装备按钮
        button_delete = QPushButton('删除装备')
        button_delete_style = """
                                            QPushButton {
                                                background-color: rgba(255, 0, 0, 0.5);
                                                border: none;
                                                border-radius: 0px;
                                                color: white;
                                                padding: 15px;
                                                font-size: 36px;
                                            }
        
                                            QPushButton:pressed {
                                                background-color: rgba(255,255,255, 1.0);
                                            }
                                            QPushButton:hover {
                                                background-color: rgba(0, 0, 0, 0.1);
                                            }
        
                                        """
        button_delete.setStyleSheet(button_delete_style)
        button_delete.setFixedSize(190, 94)
        self.layout().addWidget(button_delete, 10, 6, 1, 1)
        button_delete.clicked.connect(self.do_empty)

        # 创建咨询按钮
        button_ask = QPushButton('咨询装备')
        button_ask_style = """
                                        QPushButton {
                                            background-color: rgba(0, 255, 0, 0.5);
                                            border: none;
                                            border-radius: 0px;
                                            color: white;
                                            padding: 15px;
                                            font-size: 36px;
                                        }                     

                                        QPushButton:pressed {
                                            background-color: rgba(255,255,255, 1.0);
                                        }
                                        QPushButton:hover {
                                            background-color: rgba(0, 0, 0, 0.1);
                                        }
                 
                                    """
        button_ask.setStyleSheet(button_ask_style)
        button_ask.setFixedSize(190, 94)
        self.layout().addWidget(button_ask, 11, 6, 1, 1)
        button_ask.clicked.connect(self.do_ask)

    def do_ask(self):

        print('打开咨询界面')
        dialog = ChatDialog(self)
        dialog.show()
        print('已打开')
        # self.mask.deleteLater()  # 移除遮罩


    def send_item(self, event):
        self.open_detail_signal2.emit(self.item_data)
        print('已发送add信号')
        self.first_rate()
        print('以进行评分')
        print(self.this_rate)
        self.open_detail_signal0.emit(self.this_rate)
        print('初始评分已发送')

    def do_empty(self):
        self.open_detail_signal3.emit(self.item_data)

    def first_rate(self):
        # 创建半透明遮罩
        self.mask = QWidget(self)
        self.mask.setStyleSheet("background-color: rgba(0,0,0,0.5);")
        self.mask.setGeometry(self.rect())
        self.mask.show()
        print('正在打开')
        dialog = RatingDialog(self)
        print('已打开')
        
        if dialog.exec_() == QDialog.Accepted:  # 用户选择了评分
            score = dialog.selected_rating
            print(score)
            # 更新数据
            self.this_rate = score
            self.mask.deleteLater()  # 移除遮罩


class RatingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setWindowModality(Qt.ApplicationModal)  # 阻塞整个程序
        self.setGeometry(QRect(1150, 566, 466, 269))
        self.selected_rating = None
        print('进入打分节面1')

        layout = QVBoxLayout()
        self.setLayout(layout)

        # 提示文字
        label = QLabel("请选择评分 (1-5分)")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 40px; color: purple;")
        layout.addWidget(label)

        # 评分按钮（1-5分）
        button_layout = QHBoxLayout()
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.setFixedSize(40, 40)
            btn.clicked.connect(lambda _, x=i: self.set_rating(x))
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)

    def set_rating(self, rating):
        self.selected_rating = rating
        self.accept()  # 关闭对话框并返回 QDialog.Accepted


# 我的装备独立介绍界面
class MYITEMDETAILPAGE(QWidget):
    # 发送信号标志我的装备打分
    open_detail_signal4 = QtCore.pyqtSignal(int)
    # 发送信号表示删除本装备在我的装备中的信息
    open_detail_signal5 = QtCore.pyqtSignal(dict)

    def __init__(self, item=None, parent=None, my_rate=None):
        super().__init__(parent)
        self.item_data = item
        self.now_rate = my_rate
        self.new_rate = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.item_data.get('racket_name', '商品详情'))
        self.resize(800, 600)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 0.6);")
        self.setLayout(QGridLayout())

        # 商品信息
        title_label = QLabel(self.item_data.get('racket_name', '未命名'))
        title_label.setStyleSheet("font-size: 40px; font-weight: bold; color: black;")
        self.layout().addWidget(title_label, 0, 0, 1, 5)

        brand_label = QLabel(f"品牌: {self.item_data.get('racket_brand', '董哥')}")
        brand_label.setStyleSheet("font-size: 32px; color: black;")
        self.layout().addWidget(brand_label, 2, 0, 1, 5)

        price_label = QLabel(f"价格: ￥{self.item_data.get('racket_price_range', 0)}")
        price_label.setStyleSheet("font-size: 32px; color: red;")
        self.layout().addWidget(price_label, 4, 0, 1, 5)

        score_label = QLabel(f"我的评分: {self.now_rate}")
        score_label.setStyleSheet("font-size: 38px; color: green;")
        self.layout().addWidget(score_label, 6, 0, 1, 5)

        weight_label = QLabel(f"拍重 : {self.item_data.get('racket_weight', 0)}")
        weight_label.setStyleSheet("font-size: 32px; color: green;")
        self.layout().addWidget(weight_label, 8, 0, 1, 7)

        more_label = QLabel(f"更多信息:\n 拍杆材质: {self.item_data.get('racket_stick', '暂无描述')}"
                            f"\n 拍框材质: {self.item_data.get('racket_frame', '暂无描述')} "
                            f"\n 顶磅: {self.item_data.get('racket_max_pounds', '暂无描述')}"
                            f"\n 杆韧: {self.item_data.get('racket_toughness', '暂无描述')}"
                            f"\n 杆粗:  {self.item_data.get('racket_thickness', '暂无描述')}"
                            f"\n 杆长:  {self.item_data.get('racket_length', '暂无描述')}"
                            f"\n 平衡:  {self.item_data.get('racket_balancepoint', '暂无描述')}")
        more_label.setWordWrap(True)  # 自动换行
        more_label.setStyleSheet("font-size: 32px;")
        more_label.setAlignment(Qt.AlignLeft)
        self.layout().addWidget(more_label, 9, 0, 3, 6)

        # 创建加入按钮
        button_rate = QPushButton('更新评分')
        button_rate_style = """
                                        QPushButton {
                                            background-color: rgba(20, 20, 20, 0.5);
                                            border: none;
                                            border-radius: 0px;
                                            color: white;
                                            padding: 15px;
                                            font-size: 36px;
                                        }
            
                                        QPushButton:pressed {
                                            background-color: rgba(255,255,255, 1.0);
                                        }
                                        QPushButton:hover {
                                            background-color: rgba(0, 0, 0, 0.1);
                                        }
            
                                    """
        button_rate.setStyleSheet(button_rate_style)
        button_rate.setFixedSize(190, 94)
        self.layout().addWidget(button_rate, 9, 6, 1, 1)
        button_rate.clicked.connect(self.do_rate)

        # 创建删除装备按钮
        button_delete = QPushButton('删除装备')
        button_delete_style = """
                                    QPushButton {
                                        background-color: rgba(255, 0, 0, 0.5);
                                        border: none;
                                        border-radius: 0px;
                                        color: white;
                                        padding: 15px;
                                        font-size: 36px;
                                    }

                                    QPushButton:pressed {
                                        background-color: rgba(255,255,255, 1.0);
                                    }
                                    QPushButton:hover {
                                        background-color: rgba(0, 0, 0, 0.1);
                                    }

                                """
        button_delete.setStyleSheet(button_delete_style)
        button_delete.setFixedSize(190, 94)
        self.layout().addWidget(button_delete, 10, 6, 1, 1)
        button_delete.clicked.connect(self.do_empty)

        # 创建咨询按钮
        button_ask = QPushButton('咨询装备')
        button_ask_style = '''
                                        QPushButton {
                                            background-color: rgba(0, 255, 0, 0.5);
                                            border: none;
                                            border-radius: 0px;
                                            color: white;
                                            padding: 15px;
                                            font-size: 36px;
                                        }

                                        QPushButton:pressed {
                                            background-color: rgba(255,255,255, 1.0);
                                        }
                                        QPushButton:hover {
                                            background-color: rgba(0, 0, 0, 0.1);
                                        }

                                    '''
        button_ask.setStyleSheet(button_ask_style)
        button_ask.setFixedSize(190, 94)
        self.layout().addWidget(button_ask, 11, 6, 1, 1)
        button_ask.clicked.connect(self.do_ask)

    def do_ask(self):
        # 创建半透明遮罩
        self.mask = QWidget(self)
        self.mask.setStyleSheet("background-color: rgba(0,0,0,0.5);")
        self.mask.setGeometry(self.rect())
        self.mask.show()
        print('打开咨讯框')
        dialog = ChatDialog(self)


        print('open')

    def do_empty(self):
        self.open_detail_signal5.emit(self.item_data)

    def do_rate(self):
        # 创建半透明遮罩
        self.mask = QWidget(self)
        self.mask.setStyleSheet("background-color: rgba(0,0,0,0.5);")
        self.mask.setGeometry(self.rect())
        self.mask.show()
        print('正在打开')
        dialog = RatingDialog(self)
        print('已打开')

        if dialog.exec_() == QDialog.Accepted:  # 用户选择了评分
            score = dialog.selected_rating
            print(score)
            # 更新数据
            self.new_rate = score
            self.mask.deleteLater()  # 移除遮罩

        self.open_detail_signal4.emit(self.new_rate)
        print('已发送新的评分')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MAINWINDOW()
    window.show()
    sys.exit(app.exec_())
