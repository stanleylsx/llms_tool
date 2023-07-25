# -*- coding: utf-8 -*-
# @Time : 2023/7/2 22:05
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : make_regex.py
# @Software: PyCharm
import re


def make_regex(line):
    """
    文本变成正则字符串
    :param line:
    :return:
    """
    return re.sub(r'([()+*?\[\]])', r'\\\1', line)
