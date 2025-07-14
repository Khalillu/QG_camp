import requests

r1 = requests.get("https://cn.bing.com/search?q=requests")

post_data ={
    'stock':'000001',
    'searchkey':'',
    'category':'category_ndbg_szsh',
    'pageNum':'1',
    'pageSize':'',
    'column':'szse_main',
    'tabName':'fulltext',
    'sortName':'',
    'limit':'',
    'seDate':''
}

r2 = requests.post('https://www.cninfo.com.cn/cninfo-new/announcement/query',data=post_data)

# get方法和post方法的使用如上。这里的返回值是一个对象，这个对象包括了各种各样的属性和方法

print(r1.status_code)
print(r1.encoding)
print(r1.content)

print(r1.json) # 把请求回来的json数据转成python字典并返回

r3 = requests.get('http://www.cninfo.com.cn/finalpage/2015-03-13/1200694563.PDF'
                  ,stream=True)   # 请求
r3.raw.read(10)

def getHTML(url):
    r = requests.get(url)
    return r.content

if __name__ == '__main__':
    url = 'https://zhuanlan.zhihu.com/xmucpp'
    html = getHTML(url)
    print(html)