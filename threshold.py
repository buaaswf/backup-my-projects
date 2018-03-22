#!/usr/bin/python
#-*- coding: utf-8 -*-
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render_to_response

from django.http import HttpResponse,HttpResponseRedirect, JsonResponse

import os
import sys
import cPickle as pickle

def threshold(request):
    msg = {}
    return render_to_response('threshold.html', msg)

def post_threshold(request):
    msg = {}
    threshold = 0
    if request.GET.has_key("threshold"):
        threshold = request.GET['threshold']
        try:
            threshold = float(threshold)
        except Exception, e:
            threshold = 0
    log = request.GET.get('log', 'ours')
    path = request.GET.get('path', 'shangtang1Output_100x100_frontface_lib_obj')
    if log == '':
        log = 'ours'
    if path == '':
        path = 'shangtang1Output_100x100_frontface_lib_obj'
    page_nums = 500

    page = 1
    if request.GET.has_key("page"):
        page = int(request.GET['page'])

    correct_list, wrong_list = get_correct_wrong(threshold, log, path)

    page_total = (len(correct_list)+page_nums-1)/page_nums
    msg['pages'] = [i+1 for i in range(page_total)]

    msg['correct']  = correct_list[(page-1)*page_nums:page*page_nums]
    msg['wrong']    = wrong_list[(page-1)*page_nums:page*page_nums]
    print len(correct_list), len(wrong_list)
    if len(wrong_list) == 0:
        msg['accuracy'] = 100
    else:
        msg['accuracy'] = "%.2f"%(len(correct_list)*100.0/(len(wrong_list) + len(correct_list)))
    msg['wrong_num'] = len(wrong_list)
    msg['correct_num'] = len(correct_list)
    msg['threshold'] = threshold
    msg['path'] = path
    msg['log'] = log
    msg['range'] = range(1, 21)

    return render_to_response('threshold_show.html', msg)

def read_pkl(filename):
    f = open(filename, 'r')
    data = pickle.load(f)
    return data

def get_data(path):

    pred = {}
    with open(path,'r') as f:
        content = f.readlines()
    for imageid in content:
        [key,val] = imageid.split(',')
        key=key.split('_')[0]
        if (key):
            pred[(key)] = float(val.split('\n')[0])

    return pred

def get_correct_wrong(threshold, log, path):
    data_path = os.path.join('/home/g206/work_sdf/web-g206/find-lost/app/static/verify_result', path)

    data_url = os.path.join('/static/verify_result', path)

    same_log = os.path.join(data_path, log + '_same.txt')
    diff_log = os.path.join(data_path, log + '_diff.txt')

    same_data = get_data(same_log)
    diff_data = get_data(diff_log)

    correct = []
    wrong   = []

    for key, val in same_data.items():
        card_file = os.path.join(data_url, 'same', key + '_idcard_aligned.jpg')
        cam_file = os.path.join(data_url, 'cam', key + '_')
        if val == 0:
            continue
        if val >= threshold:
            correct.append([val, card_file, cam_file, key])
        else:
            wrong.append([val, card_file, cam_file, key])

    for key, val in diff_data.items():
        card_file = os.path.join(data_url, 'diff', key + '_idcard_aligned.jpg')
        cam_file = os.path.join(data_url, 'cam', key + '_')
        if val == 0:
            continue
        if val >= threshold:
            wrong.append([val, card_file, cam_file, key])
        else:
            correct.append([val, card_file, cam_file, key])

    return correct, wrong
