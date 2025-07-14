import requests
from bs4 import BeautifulSoup

# 调用get函数，将网页请求下来
r = requests.get('http://www.wise.xmu.edu.cn/people/faculty')
# r是一个包含了整个HTTP协议所需要的各种各样的东西的对象。
html = r.content
# 查看源文件，查看标签

# 创建一个BeautifulSoup对象
soup = BeautifulSoup(html, 'html.parser')
# html.parser是解析器

# 首先提取这部分代码的第一行，定位到这部分代码
# 使用BeaurifulSoup对象的find方法，这个方法的意思是找到带有‘div’这个标签
# 并且参数包含“class='people_list'”的HTML代码，如果有多个的话，find方法就取第一个。
# 现在，我们要取出所有“a”标签里面的内容
# find方法里面也可以加上attrs参数，然后用字典传入我们想找的标签属性。
div_people_list = soup.find('div', attrs={'class':'people_list'})

# 使用find_all方法取出所有标签为“a”并且参数包含“target = '_blank'”的代码，返回一个列表。
# "a"标签里面的"href"参数是我们需要的老师个人主页的信息，而标签里面的文字是老师的名字。
# attrs参数包含{...}
a_s = div_people_list.find_all('a', attrs={'target':'_blank'})

# 把a标签里面的"href"参数的值提取出来，赋值给url，使用get_text()提取文字
for a in a_s:
    url = a['href']
    name = a.get_text()
    print(name,url)

