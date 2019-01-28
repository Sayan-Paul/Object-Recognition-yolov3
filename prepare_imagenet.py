#! /usr/bin/env python
"""ImageNet dataset processing"""

import mmap
from tqdm import tqdm
import sys
import os
import time
import tarfile

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse
    import urllib

from utils import get_immediate_subdirectories, get_file_list
from collections import defaultdict

__author__ = "Sayan Paul"
__email__ = "sayanpau@usc.edu"


objects = "bell pepper,bottle,broccoli,brush,butter,cheese,corn,cream cheese,croutons," \
          "flour,honey,lettuce,measuring cup,pasta,pot,spaghetti,sugar,water,yogurt".split(',')
obj_map = {
    'apple': ' n07739125',
    ' blender': ' n02850732',
    ' bowl': ' n02880940',
    ' bottle': ' n02876657',
    ' bread': ' n07679356',
    ' broccoli': ' n07714990',
    ' brush': ' n02908217',
    ' butter': ' n07848338',
    ' carrot': ' n07730207',
    ' cheese': ' n07850329',
    ' chicken': ' n07592317',
    ' chocolate': ' n04972451',
    ' corn': ' n12144580',
    ' cream cheese': ' n07851298',
    ' croutons': ' n07682197',
    ' cucumber': ' n07718472',
    ' measuring cup': ' n03733805',
    ' dough': ' n07861158',
    ' doughnut': ' n07639069',
    ' egg': ' n07840804',
    ' fish': ' n02512938',
    ' flour': ' n07569106',
    ' fork': ' n02973805',
    ' honey': ' n07858978',
    ' jelly': ' n07643981',
    ' knife': ' n03623556',
    ' lemon': ' n07749582',
    ' lettuce': ' n07723559',
    ' milk': ' n07844604',
    ' mustard': ' n07819480',
    ' onion': ' n07722217',
    ' pan': ' n03880323',
    ' peanut butter': ' n07855510',
    ' pasta': ' n07863374',
    ' pepper grinder': ' n03914337',
    ' bell pepper': ' n07720875',
    ' pitcher': ' n03950228',
    ' plate': ' n07579787',
    ' pot': ' n03990474',
    ' rolling pin': ' n04103206',
    ' salt': ' n07813107',
    ' pepper shaker': ' n03914438',
    ' spatula': ' n04270147',
    ' spaghetti': ' n07700003',
    ' spoon': ' n04284002',
    ' spreader': ' n04287986',
    ' steak': ' n07877961',
    ' sugar': ' n07859284',
    ' tomato': ' n07734017',
    ' pasta sauce': ' n07838233',
    ' tongs': ' n04450749',
    ' turkey': ' n07678953',
    ' water': ' n04559910',
    ' whisk': ' n04578934',
    ' yogurt': ' n07849336'
}


def get_annotation_map(data_directory):
    """

    :param data_directory:
    :type data_directory:
    :return:
    :rtype:
    """

    new_obj_map = {k.strip(): v.strip() for k,v in obj_map.items()}

    concerned_objs = [ new_obj_map[k] for k in new_obj_map if k in objects]

    result = {}
    obj_dir_list = get_immediate_subdirectories(data_directory)
    for obj_dir in obj_dir_list:
        obj = os.path.basename(obj_dir)
        if obj not in concerned_objs:
            continue
        result[obj] = []
        anno_dir = os.path.join(obj_dir, 'Annotation')
        for anno in get_file_list(anno_dir, format=".xml"):
            anno_name = os.path.splitext(os.path.basename(anno))[0]
            result[obj].append(anno_name)
    return result


if __name__ == "__main__":

    data_dir = "data/ImageNet"
    anno_map = get_annotation_map(data_dir)
    with open(os.path.join(data_dir, 'select_urls.txt'), 'r', encoding='ISO-8859-1') as url_file:
        url_data = url_file.readlines()
    url_map = dict()
    name_map = dict()
    for line in url_data:
        obj, count, url = line.split()
        url_map[obj + "_" + count] = url
        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        name_map[os.path.basename(path)] = obj + "_" + count

    for k in anno_map:
        row = []
        for label in anno_map[k]:
            if label in url_map:
                row.append(url_map[label])
        anno_map[k] = row

    for k in anno_map:
        for url in anno_map[k]:
            try:
                scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
                filename = os.path.join(data_dir, 'downloaded_images', name_map[os.path.basename(path)] + ".jpg")
                print(filename)
                if os.path.exists(filename):
                    continue
                u = urllib2.urlopen(url)
                with open(filename, 'wb') as f:
                    meta = u.info()
                    meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
                    meta_length = meta_func("Content-Length")
                    file_size = None
                    if meta_length:
                        file_size = int(meta_length[0])
                    print("Downloading: {0} Bytes: {1}".format(url, file_size))

                    file_size_dl = 0
                    block_sz = 8192
                    while True:
                        buffer = u.read(block_sz)
                        if not buffer:
                            break

                        file_size_dl += len(buffer)
                        f.write(buffer)

                        status = "{0:16}".format(file_size_dl)
                        if file_size:
                            status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                        status += chr(13)
            except:
                print('Fail to download : ' + url)
