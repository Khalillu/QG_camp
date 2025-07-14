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

    for company_ul in company_list_ct.find_all('ul'
            ,attrs = {'class':'company-list'}):
        for company_li in company_ul.find_all('li'):
            company_url = company_li.a['href']
            company_info = company_li.get_text()
            print(company_info,company_url)
if __name__ == '__main__':
    URL = 'https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/#id9'
    html = getHTML(URL)
    parseHTML(html)