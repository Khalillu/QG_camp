from DrissionPage import ChromiumPage
import pandas as pd
from tqdm import tqdm
import time


def get_info():
    global i
    # 页面滚动到底部，方便查看爬到第几页
    time.sleep(2)
    page.scroll.to_bottom()
    # 定位包含学校信息的div
    divs = page.eles('tag:div@class=school-search_schoolItem__3q7R2')
    # 提取学校信息
    for div in divs:
        # 提取学校名称
        school = div.ele('.school-search_schoolName__1L7pc')
        school_name = school.ele('tag:em')
        # 提取学校城市
        city = div.ele('.school-search_cityName__3LsWN')
        if len(city.texts()) == 2:
            city_level1 = city.texts()[0]
            city_level2 = city.texts()[1]
        elif len(city.texts()) == 1:
            city_level1 = city.texts()[0]
            city_level2 = ""
        else:
            city_level1 = ""
            city_level2 = ""
        # 提取学校标签
        tags = div.ele('.school-search_tags__ZPsHs')
        spans = tags.eles('tag:span')
        spans_list = []
        for span in spans:
            spans_list.append(span.text)

        # 信息存到contents列表
        contents.append([school_name.text, city_level1, city_level2, spans_list])
        # print(school_name.text, city.text, spans_list)
    print("爬取第", i, "页，总计获取到", len(contents), "条大学信息")

    time.sleep(2)

    # 定位下一页，点击下一页
    try:
        next_page = page.ele('. ant-pagination-next')
        next_page.click()
    except:
        pass


def craw():
    global i
    for i in tqdm(range(1, 146)):
        # 每爬50页暂停1分钟
        if i % 50 == 0:
            get_info()
            print("暂停1分钟")
            time.sleep(60)
        else:
            get_info()


def save_to_csv(data):
    # 保存到csv文件
    name = ['school_name', 'city_level1', 'city_level2', 'tags']
    df = pd.DataFrame(columns=name, data=data)
    df.to_csv(f"高考网大学信息{len(data)}条.csv", index=False)
    print("保存完成")


if __name__ == '__main__':
    # contents列表用来存放所有爬取到的大学信息
    contents = []
    # 打开网页
    page = ChromiumPage()
    page.get('https://www.gaokao.cn/school/search')

    # 声明全局变量i
    i = 0

    craw()

    save_to_csv(contents)