#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author:lch02
@Time:  2017/12/25 14:32
@Description: 
"""
from deal_files import export_data
from deal_files import bag_of_words
__author__ = 'lch02'

if __name__ == '__main__':
    # export_data()
    print  bag_of_words(['hello','i','love','u'])
    """
    import re
    sentence = '#le @lee i love u @ ooooo #le # leeee'
    sentence = re.sub(r'@\s*[\w]+ | ?#[\w]+', '', sentence)
    print sentence
    """