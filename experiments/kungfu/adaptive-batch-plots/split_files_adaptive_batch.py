import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import re

import datetime

from collections import defaultdict
import math
import random


# Regex used to match relevant loglines (in this case, a specific IP address)
line_regex = re.compile(r".*$")

def isStart(s, train=True):
    pattern = re.compile(r"\[BEGIN\sTRAINING\sKEY\]\s(?P<name>.*)", re.VERBOSE)
    if not train:
       pattern = re.compile(r"\[BEGIN\sVALIDATION\sKEY\]\s(?P<name>.*)", re.VERBOSE)

    match = pattern.match(s)
    if match is None:
        return False
    return True

def isEnd(s, train=True):
    pattern = re.compile(r"\[END\sTRAINING\sKEY\]\s(?P<name>.*)", re.VERBOSE)
    if not train:
        pattern = re.compile(r"\[END\sVALIDATION\sKEY\]\s(?P<name>.*)", re.VERBOSE)

    match = pattern.match(s)
    if match is None:
        return False
    return True


def write(file, line):
    with open(file, "a") as myfile:
        myfile.write(line)




def split(log_file, train=True):
    with open(log_file, "r") as in_file:
        startFlag = False
        filename = None
        lines = []
        for i, line in enumerate(in_file):
            if (line_regex.search(line)):
                start = isStart(line, train=train)
                if start:
                    startFlag = True
                    filename = line.split(" ")[3]
                if  startFlag:  
                    lines.append(line)
                end = isEnd(line, train=train)
                if end:
                    startFlag = False
                    for l in lines:
                            write(filename.rstrip("\n\r"), l)
                    filename = None

split("resnet-32-adaptive-batch-0.01-AFTERNOON-REPRODUCIBLE-NO-WARMUP.log", train=True)
split("resnet-32-adaptive-batch-0.01-AFTERNOON-REPRODUCIBLE-NO-WARMUP.log", train=False)