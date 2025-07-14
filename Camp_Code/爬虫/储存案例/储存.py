import pandas as pd
import csv
import codecs
# -*- coding: utf-8 -*-
# python列表or元组与csv的转换

# csv.reader(file)    # 读出csv文件
# csv.writer(file)    # 写入csv文件
# writer.writerow(data)   # 写入一行数据
# writer.writerows(data)  # 写入多行数据
#
# # Python字典与csv的转换
# csv.DictReader(file) # 读出csv文件
#
# csv.DictWriter(file) # 写入csv文件
# writer.writeheader()    # 写文件头
# writer.writerow(data)   # 写入一行数据
# writer.writerows(data)  # 写入多行数据

import requests
from bs4 import BeautifulSoup

# 在审查元素里面去找我们需要的数据，接下来要做的事情是，
# 用bs4这个强大的工具，通过前面提到的标签和标签的属性定位到某个标签。

def getHTML(url):
    r = requests.get(url)
    return r.content
# parser是解析器。
def parseHTML(html):
    soup = BeautifulSoup(html,'html.parser')
    # HTML的标签习惯用“-”连接，比如“list-ct”，而不是“_”，比如
    body=soup.body
    company_middle = body.find('div',
                               attrs={'class':'middle'})
    company_list_ct = company_middle.find('div',
                                attrs={'class':'list-ct'})
    company_list = []

    for company_ul in company_list_ct.find_all('ul'
            ,attrs = {'class':'company-list'}):
        for company_li in company_ul.find_all('li'):
            company_url = company_li.a['href']
            company_info = company_li.get_text()
            company_list.append([company_info.encode('utf-8'),company_url.encode('utf-8')])

    return company_list

def writeCSV(file_name,data_list):
    with codecs.open(file_name,'wb') as f:
        writer = csv.writer(f)
        for data in data_list:
            writer.writerow(data)

if __name__ == '__main__':
    URL = 'https://cycling.today/2025-tour-de-france-live-stream/'
    html = getHTML(URL)
    data_list = parseHTML(html)
    writeCSV('test.csv',data_list)