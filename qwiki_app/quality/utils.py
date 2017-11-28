# coding: utf-8
import re
import os
import time
import logging
import hashlib
import unicodedata
import multiprocessing
from collections import Counter


__author__ = 'shiyue'
"""
some util functions
"""

PAT_ALPHABETIC = re.compile('(([A-Za-z0-9])+)', re.UNICODE)


def count_revisions(revision_file):
    """count revision_file and return page distribution on class"""
    class_list = map(lambda x: x.split('\t'), open(revision_file, 'r'))
    class_list = map(lambda x: (x[0], x[1]), class_list)
    class_list = [k for k, _ in Counter(class_list).keys()]
    return Counter(class_list)


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def tokenize(text):
    """convert text to word list"""
    text = any2unicode(text)
    text = text.lower()
    text = deaccent(text)
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def text_to_words(text):
    """convert text to word list, remove stopwords and return word_list string"""
    # stopwords = get_stopwords()
    words = []
    for word in tokenize(text):
        # if word not in stopwords and len(word) > 1:
            words.append(word)
    return ' '.join(words)


def deaccent(text):
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.
    Return input string with accents removed, as unicode.
    deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'
    """
    if not isinstance(text, unicode):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = u''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def get_stopwords():
    """return stopwords set"""
    return set(open("/home/zhangshiyue/paper_work_wikipedia/get_data/stopwords", 'r').readline().split(' '))


def get_length_LCS(x_words, y_words):
    """calculate the longest common sequence of two word list"""
    m = len(x_words)
    n = len(y_words)
    if m == 0 or n == 0:
        return []
    c = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x_words[i - 1] == y_words[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i - 1][j], c[i][j - 1])
    return c


def read_out_LCS(c, x_words, y_words, i, j):
    """
    read out longest common sequence from i, j
    return a list of tuples, there are three elements in each tuple:
    (word pos in x_words,word pos in y_words, word)
    """
    if not c:
        return []
    res = []
    while i != 0 and j != 0:
        if x_words[i - 1] == y_words[j - 1]:
            res.append((i - 1, j - 1, x_words[i - 1]))
            i -= 1
            j -= 1
        else:
            if c[i][j - 1] >= c[i - 1][j]:
                j -= 1
            else:
                i -= 1
    return res


def log_config(name):
    """log config"""
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(filename)s[line: %(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H: %M: %S',
            filename="/extdisk/student/zhangshiyue/{}.log".format(name),
            filemode="a")
    return logging.getLogger(name)


def user_name_to_id_map():
    """
    return a map of user name -> inside id
    id is str
    """
    edict = {}
    for i, line in enumerate(open("/extdisk/wtist/zhangshiyue/editors_info", 'r')):
        edict[line.split('\t')[0]] = str(i)
    return edict


def user_id_to_name_map():
    name_id = user_name_to_id_map()
    id_name = {}
    for name, id in name_id.iteritems():
        id_name[id] = name
    return id_name


def page_title_to_id_map():
    """return a map of page title -> id"""
    pdict = {}
    for i, line in enumerate(open("/extdisk/wtist/zhangshiyue/history_titles", 'r')):
        items = line[:-1].split('\t')
        pdict[items[2]] = items[1]
    return pdict


def md5(content):
    m = hashlib.md5()
    m.update(content)
    return m.hexdigest()


def timestamp_to_second(timestamp, str_format="%Y-%m-%dT%H:%M:%SZ"):
    return time.mktime(time.strptime(timestamp, str_format))


def mutiprocess_pages_and_save(input_dir, output_dir, process_num, process_fun, logger=None):
    """
    using multiprocess process each page and save
    process_num is the num of multi process
    process_fun is the function to process each page
    """
    pages = multiprocessing.Queue()
    for p in os.listdir(input_dir):
        pages.put(p)

    class MyProcess(multiprocessing.Process):
        def __init__(self, i):
            multiprocessing.Process.__init__(self, name="Process-%d" % i)
            self.daemon = True
            self.process = process_fun

        def run(self):
            while not pages.empty():
                page = pages.get()
                start = time.time()
                self.process(input_dir, output_dir, page)
                end = time.time()
                if logger:
                    logger.info("{}:{}".format(page, (end - start) / 60))
                else:
                    print "{}:{}".format(page, (end - start) / 60)

    n = process_num
    threads = []
    for i in range(n):
        threads.append(MyProcess(i))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == '__main__':
    print count_revisions("/extdisk/wtist/zhangshiyue/history_titles")
