#!/usr/bin/python
#-*- coding: utf-8 -*-
from django.shortcuts import render
from django.http import HttpResponse

from django.http import HttpResponse,HttpResponseRedirect, JsonResponse
from django.shortcuts import render_to_response
from django.template import RequestContext
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_protect
from django.template.context_processors import csrf
from django.template import loader

from django.core.paginator import Paginator, InvalidPage, EmptyPage, PageNotAnInteger
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from os.path import normpath, join
from website import settings

from forms import *
from models import *
import time
import os
import sys

HOMEPAGE = "http://115.25.160.254:9999"

from face_verify_client import *
from common import *


@csrf_exempt
def verify_ui(request):
    msg = {}
    return render_to_response('verify.html', msg)

@csrf_exempt
def verify_post(request):
    result = -1
    try:
        image_left  = request.FILES['localFile_left']
        image_right = request.FILES['localFile_right']


        left_file  = '/static/verify_imgs/' + getRandomStr() + image_left.name[-4:]
        right_file = '/static/verify_imgs/' + getRandomStr() + image_right.name[-4:]

        left_path = settings.APP_PATH + '/app' + left_file
        right_path = settings.APP_PATH + '/app' + right_file

        handle_uploaded_photo(image_left, left_path)
        handle_uploaded_photo(image_right, right_path)

        result = face_verify(left_path, right_path)

        #result = 0.5
        if result[0] == 'C':
            return render_to_response('back.html',{'Similar':0, "Result":result})

        print JsonResponse({'Similar':result})
        r = "unknown"
        if float(result) >= 0.5:
            r = "Same person"
        else:
            r = "Different person"

        r = ""

        left_file = left_file.split('.')[0] + '_drawed.' + left_file.split('.')[1]
        print "left", left_file
        right_file = right_file.split('.')[0] + '_drawed.' + right_file.split('.')[1]
        print "right", right_file
        return render_to_response('back.html',{'Similar':result, "Result":r, 'left_path':left_file, 'right_path':right_file })
        #return HttpResponse('<p>' + str(result) + '</p>')

    except Exception, e:
        print str(e)
        return render_to_response('back.html',{'Similar':result})



def handle_uploaded_photo(f, filepath):
    with open( filepath, 'wb+') as info:

        for chunk in f.chunks():
            info.write(chunk)
