# coding: utf8
import re
import json
import urllib
import time
import numpy
import theano
import utils

__author__ = 'shiyue'

"""
利用维基API获取数据
"""
BASEURL = "https://en.wikipedia.org/w/api.php?"


def get_data(pid):
    means = [1264008714.139915, 3509179.8353327094, 0.2504408702260013, 43.06438230327548, 0.012182197405231923,
             44.56629459686197, 538695.5362061551, 171608286.81868827, 0.5128915757461078, 675.2385349412635,
             0.7896423034454816, 1738.223383152001, 294.0369398338928, 69.59082089934849, 41269.406682300105,
             243.97608756207467, 0.8980337694577826, 3.4076605385797776, 49.220079229609446, 30.670648110745113,
             34.660347347153504, 8.41828439909491, 0.6541052478268352, 10.711160428692862, 6.589008565301402]

    stds = [98388755.42487112, 5574705.419802753, 0.43326693931971394, 50.30835530337637, 0.10969863932767004,
            10519.50492344086, 3220230.7403492616, 110992198.0905063, 0.1547291416132058, 930.341999542491,
            0.6347494622366526, 2578.656425519516, 1338.637527966289, 197.96642050340265, 36170.225333237235,
            205.87548419319728, 0.04984624284215657, 29.17096178688342, 63.532406800927255, 50.76459843660398,
            66.11081081864211, 8.35105017644952, 0.47565909282738567, 13.587859172594351, 6.9228341506182005]

    query_for_text = {
        "action": "query",
        "prop": "revisions",
        "format": "json",
        "rvprop": "content",
        "rvlimit": 1,
    }
    query = {
        "action": "query",
        "prop": "revisions",
        "format": "json",
        "rvprop": "ids|flags|timestamp|user|userid|size|sha1|comment",
        "rvlimit": 50,
    }
    query["titles"] = pid
    query_for_text["titles"] = pid

    url = BASEURL + urllib.urlencode(query_for_text)
    res = json.loads(urllib.urlopen(url).read())
    pid = res["query"]["pages"].keys()[0]
    body = res["query"]["pages"][pid]
    revision = body["revisions"][0]
    content = revision["*"].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') \
        if "*" in revision else ""

    x_text = numpy.zeros((1, 11)).astype(theano.config.floatX)
    article_length = len(content)
    words = len(utils.text_to_words(content))
    numimages = len(_find_pattern(content, "\[\[Image:[^\]]*]]"))
    numrefs = len(_find_pattern(content, "<\/ref>"))
    num2sections = len(_find_pattern(content, "[^=](==[^=]*==)"))
    num3sections = len(_find_pattern(content, "[^=](===[^=]*===)"))
    infobox, templates, cite_templates = _get_templates_simple(content)
    categories, numlinks = __get_links_categories(content)
    x_text[0][1] = (numlinks - means[15]) / stds[15]
    x_text[0][2] = (float(words) / article_length - means[16]) / stds[16]
    x_text[0][3] = (numimages - means[17]) / stds[17]
    x_text[0][4] = (numrefs - means[18]) / stds[18]
    x_text[0][5] = (cite_templates - means[19]) / stds[19]
    x_text[0][6] = (templates - cite_templates - means[20]) / stds[20]
    x_text[0][7] = (categories - means[21]) / stds[21]
    x_text[0][8] = (infobox - means[22]) / stds[22]
    x_text[0][9] = (num2sections - means[23]) / stds[23]
    x_text[0][10] = (num3sections - means[24]) / stds[24]

    url = BASEURL + urllib.urlencode(query)
    res = json.loads(urllib.urlopen(url).read())
    body = res["query"]["pages"][str(pid)]
    revisions = body["revisions"]

    x = numpy.zeros((50, 1, 6)).astype(theano.config.floatX)
    x_mask = numpy.zeros((50, 1)).astype(theano.config.floatX)

    i = 0
    for revision in revisions[::-1]:
        x_mask[i][0] = 1.0
        minor = 1 if "minor" in revision else 0
        userid = int(revision["userid"]) if "userid" in revision else 0
        timestamp = utils.timestamp_to_second(revision["timestamp"] if "timestamp" in revision else "")
        size = int(revision["size"]) if "size" in revision else 0
        comment = revision["comment"].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') \
            if "comment" in revision else ""
        comlen = len(comment)
        revert = 1 if filter(lambda w: w == 'rv' or w == 'revert', comment.split(' ')) != [] else 0
        x[i][0][0] = (timestamp - means[0]) / stds[0]
        x[i][0][1] = (userid - means[1]) / stds[1]
        x[i][0][2] = (minor - means[2]) / stds[2]
        x[i][0][3] = (comlen - means[3]) / stds[3]
        x[i][0][4] = (revert - means[4]) / stds[4]
        x[i][0][5] = (size - means[14]) / stds[14]
        i += 1

    x_text[0][0] = (size - means[14]) / stds[14]

    return x, x_mask, x_text


def _find_pattern(content, pattern):
    obj = re.compile(pattern)
    return obj.findall(content)


def _get_templates_simple(content):
    templates = _find_pattern(content, "{{.*?}}")
    isinfo = 0
    numcite = 0
    for temp in templates:
        if 'infobox' in temp.lower():
            isinfo = 1
        if 'cite' in temp.lower():
            numcite += 1
    return isinfo, len(templates), numcite


def __get_links_categories(content):
    links = _find_pattern(content, "\[\[.*?\]\]")
    numcat = 0
    for link in links:
        if 'category:' in link.lower():
            numcat += 1
    return numcat, len(links)


def _get_links(rid):
    query = {
        "action": "query",
        "prop": "links",
        "pllimit": 500,
        "format": "json",
        "revids": rid
    }
    nlinks = 0
    while True:
        url = BASEURL + urllib.urlencode(query)
        try:
            res = json.loads(urllib.urlopen(url).read())
        except:
            time.sleep(60)
            continue
        pages = res["query"]["pages"]
        for page, links in pages.iteritems():
            nlinks += len(links["links"])
        if "continue" in res:
            value = res["continue"]["plcontinue"]
            query["plcontinue"] = value.encode('utf-8')
        else:
            break
    return nlinks


def _get_templates(rid):
    query = {
        "action": "query",
        "prop": "templates",
        "tllimit": 500,
        "format": "json",
        "revids": rid
    }
    ntemplates = []
    while True:
        url = BASEURL + urllib.urlencode(query)
        try:
            res = json.loads(urllib.urlopen(url).read())
        except:
            time.sleep(60)
            continue
        pages = res["query"]["pages"]
        for page, templates in pages.iteritems():
            for template in templates["templates"]:
                ntemplates.append(template["title"])
        if "continue" in res:
            value = res["continue"]["tlcontinue"]
            query["tlcontinue"] = value.encode('utf-8')
        else:
            break
    cite_templates = [tem for tem in ntemplates if
                      'Cite' in tem or 'Citation' in tem or 'cite' in tem or 'citation' in tem]
    infobox = [tem for tem in ntemplates if 'Infobox' in tem or 'infobox' in tem] != []
    return len(ntemplates), len(cite_templates), infobox


def _get_categories(rid):
    query = {
        "action": "query",
        "prop": "categories",
        "cllimit": 500,
        "format": "json",
        "revids": rid
    }
    ncatagories = 0
    while True:
        url = BASEURL + urllib.urlencode(query)
        try:
            res = json.loads(urllib.urlopen(url).read())
        except:
            time.sleep(60)
            continue
        pages = res["query"]["pages"]
        for page, categories in pages.iteritems():
            ncatagories += len(categories["categories"])
        if "continue" in res:
            value = res["continue"]["clcontinue"]
            query["clcontinue"] = value.encode('utf-8')
        else:
            break
    return ncatagories


if __name__ == '__main__':
    # ts = []
    # import random
    #
    # for line in open("lstm_plan/data/test.txt"):
    #     p = random.random()
    #     if p < 0.01:
    #         pid = line.strip().split('\t')[0]

    # start = time.time()
    # t = get_data("Seminoma")
    # end = time.time()
    # print end - start
    #         ts.append(t)
    # print sum(ts) / len(ts)
    # print t
    pass
