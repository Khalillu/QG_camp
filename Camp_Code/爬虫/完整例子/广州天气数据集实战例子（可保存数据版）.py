from DrissionPage import ChromiumPage
import pandas as pd
from tqdm import tqdm
import time

page = ChromiumPage()
page.get('http://www.tqyb.com.cn/')


def get_info():
    # 页面滚动到底部，方便查看爬到第几页
    time.sleep(2)
    # 定位包含天气信息的ul


    # 提取温度、雨量、风速信息

    update_time = page.ele('#wind-date').text.replace('数据更新时间：', '')

    temp_data = []
    rain_data = []
    wind_data = []

    tbody = page.ele('#statistics-temp')

    for tr in tbody.eles('tag:tr')[1:]:
        cells = tr.eles('tag:td')
        if len(cells) >= 2:  # 确保有区域名和风速数据
            area = cells[0].text  # 区域名称
            temp1 = cells[1].text  # 温度值
            temp2 = cells[2].text
            temp3 = cells[3].text
            temp4 = cells[4].text
            temp5 = cells[5].text
            temp_data.append({
                '区域': area,
                '温度1': temp1,
                '温度2': temp2,
                '温度3': temp3,
                '温度4': temp4,
                '温度5': temp5,
                '更新时间': update_time
            })

    tbody = page.ele('#statistics-rain')

    # 提取温度、雨量、风速信息

    for tr in tbody.eles('tag:tr')[1:]:
        cells = tr.eles('tag:td')
        if len(cells) >= 2:  # 确保有区域名和风速数据
            area = cells[0].text  # 区域名称
            rain1 = cells[1].text  # 雨量值
            rain2 = cells[2].text
            rain3 = cells[3].text
            rain4 = cells[4].text
            rain5 = cells[5].text
            rain_data.append({
                '区域': area,
                '雨量1': rain1,
                '雨量2': rain2,
                '雨量3': rain3,
                '雨量4': rain4,
                '雨量5': rain5,
                '更新时间': update_time
            })

    tbody = page.ele('#statistics-wind')

    for tr in tbody.eles('tag:tr')[1:]:
        cells = tr.eles('tag:td')
        if len(cells) >= 2:  # 确保有区域名和风速数据
            area = cells[0].text  # 区域名称
            wind_speed1 = cells[1].text  # 风速值
            wind_speed2 = cells[2].text
            wind_speed3 = cells[3].text
            wind_speed4 = cells[4].text
            wind_speed5 = cells[5].text
            wind_data.append({
                '区域': area,
                '风速1': wind_speed1,
                '风速2': wind_speed2,
                '风速3': wind_speed3,
                '风速4': wind_speed4,
                '风速5': wind_speed5,
                '更新时间': update_time
            })

    return [[x, y, z] for x, y, z in zip(temp_data, rain_data, wind_data)]


def save_to_csv(data):
    # Transform the nested data into a flat format
    flat_data = []
    for item in data:
        temp, rain, wind = item
        flat_row = {
            'city_area': temp['区域'],
                '0~5℃温度': temp['温度1'],
                    '5~35℃温度': temp['温度2'],
            '35~37℃温度': temp['温度3'],
            '37~39℃温度': temp['温度4'],
            '39~℃温度': temp['温度5'],
            '降水量0~10mm': rain['雨量1'],
            '降水量10~20mm': rain['雨量2'],
            '降水量20~30mm': rain['雨量3'],
            '降水量30~40mm': rain['雨量4'],
            '降水量40~50mm': rain['雨量5'],
            '风速: 0~10.8m/s': wind['风速1'],
            '风速: 10.8~17.2m/s': wind['风速2'],
            '风速: 17.2~24.5m/s': wind['风速3'],
            '风速: 24.5~32.7m/s': wind['风速4'],
            '风速: 32.7m/s~': wind['风速5'],
            'tags': temp['更新时间']  # Assuming all update times are the same
        }
        flat_data.append(flat_row)

    # Define column names in the correct order
    columns = [
        'city_area',
        '0~5℃温度', '5~35℃温度', '35~37℃温度', '37~39℃温度', '39~℃温度',
        '降水量0~10mm', '降水量10~20mm', '降水量20~30mm', '降水量30~40mm', '降水量40~50mm',
        '风速: 0~10.8m/s', '风速: 10.8~17.2m/s', '风速: 17.2~24.5m/s', '风速: 24.5~32.7m/s', '风速: 32.7m/s~',
        'tags'
    ]

    # Create DataFrame and save to CSV
    df = pd.DataFrame(flat_data, columns=columns)
    filename = f"广州天气数据集{len(flat_data)}条.csv"
    df.to_csv(filename, index=False, encoding='utf_8_sig')  # Using utf_8_sig for Chinese characters
    print(f"成功保存到 {filename}")

data = get_info()

if data:
    df = pd.DataFrame(data)
    df.to_csv('TempRainWind_data.csv', index=False)
    print(f"成功保存{len(data)}条温度、雨量和风速数据")
    for item in data:
        print(f"地区：{item[0]['区域']},0~5℃温度: {item[0]['温度1']}, 5~35℃温度：{item[0]['温度2']},"
              f" 35~37℃温度：{item[0]['温度3']}, 37~39℃温度：{item[0]['温度4']},39℃~：{item[0]['温度5']},"
              f"雨量: 降水量0~10mm：{item[1]['雨量1']},雨量: 降水量10~20mm：{item[1]['雨量2']},"
              f"雨量: 降水量20~30mm：{item[1]['雨量3']},雨量: 降水量30~40mm：{item[1]['雨量4']},雨量: 降水量40~mm：{item[1]['雨量5']},"
              f"风速: 0~10.8m/s：{item[2]['风速1']},风速: 10.8~17.2m/s：{item[2]['风速2']},风速: 17.2~24.5m/s：{item[2]['风速3']},风速: 24.5~32.7m/s：{item[2]['风速4']},风速: 32.7~m/s：{item[2]['风速5']}")
else:
    print("未获取到数据")


save_to_csv(data)

# 关闭浏览器
page.close()


