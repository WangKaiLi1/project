import datetime


def file_save_time():
    current_time = datetime.datetime.now()
    # 格式化当前时间为字符串
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S')
    # 打印格式化后的当前时间
    return formatted_time
