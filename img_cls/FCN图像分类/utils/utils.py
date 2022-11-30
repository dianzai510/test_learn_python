import os

"""
获取按时间排序的最后一个文件
"""
def getlastfile(path):
    list_file = [path + '/' + f for f in os.listdir(path)]
    if len(list_file) > 0:
        list_file.sort(key=lambda fn: os.path.getmtime(fn))
        return list_file[-1]
    else:
        return None


if __name__ == '__main__':
    a = getlastfile('D:/桌面/JP')
    pass