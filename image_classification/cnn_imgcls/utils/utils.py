import os

"""
获取按时间排序的最后一个文件
"""


def getlastfile(path, ext):
    if os.path.exists(path) is not True: return None
    list_file = [path + '/' + f for f in os.listdir(path) if f.endswith(".pth")]  # 列表解析
    if len(list_file) > 0:
        list_file.sort(key=lambda fn: os.path.getmtime(fn))
        return list_file[-1]
    else:
        return None


if __name__ == '__main__':
    a = getlastfile('D:/work/proj/xray/test_learn_python/image_classification/cnn_imgcls/run/train/oqa_agl/weights',
                    '.pth')
    pass
