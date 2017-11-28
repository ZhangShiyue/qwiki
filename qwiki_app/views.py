# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from quality import feature, qmodel, qmodel_reg
from django.conf import settings
import time


def index(request):
    return render(request, 'index.html')


def api(request):
    start = time.clock()
    url = request.GET['url']
    title = url.split('/')[-1].replace('_', ' ')
    input = feature.get_data(title)
    quality = qmodel.get_quality(input)
    elapsed = round((time.clock() - start), 2)
    result = "Title: %s, Quality: %s, Used Time: %s s" % (title, quality, elapsed)
    return HttpResponse(result)


def assess(request):
    start = time.clock()
    atitle = request.GET['atitle']
    input = feature.get_data(atitle)
    quality = qmodel.get_quality(input)
    elapsed = round((time.clock() - start), 2)
    return render(request, 'index.html', {"title": atitle, "result": quality, "use_time": elapsed})
