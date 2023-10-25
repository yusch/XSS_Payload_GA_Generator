import os
import sys
import random
import numpy as np
import codecs
import configparser
import pandas as pd
import locale
import time
import subprocess
import re
from decimal import Decimal
from util import Utilty
from tqdm import tqdm
import datetime
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import waf

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

# Genetic algorithm.


class GeneticAlgorithm:

    def __init__(self, template, browser, idx, payload_list):
        self.util = Utilty()
        self.template = template
        self.obj_browser = browser

        self.idx = idx
        self.payload_list = payload_list

        # Read config.ini
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(full_path, 'config.ini'))
        except FileExistsError as e:
            self.util.print_exception(e, 'File exists error: {}'.format(e))
            sys.exit(1)

        # Common settings.
        self.wait_time = float(config['Common']['wait_time'])
        self.html_dir = self.util.join_path(
            full_path, config['Common']['html_dir'])
        self.html_template = config['Common']['html_template']
        self.html_template_path = self.util.join_path(
            self.html_dir, self.html_template)
        self.html_file = config['Common']['ga_html_file']
        self.result_dir = self.util.join_path(
            full_path, config['Common']['result_dir'])

        # Genetic algorithm settings.
        self.genom_length = int(config['Genetic']['genom_length'])
        self.max_genom_list = int(config['Genetic']['max_genom_list'])
        self.select_genom = int(config['Genetic']['select_genom'])
        self.individual_mutation_rate = float(
            config['Genetic']['individual_mutation_rate'])
        self.genom_mutation_rate = float(
            config['Genetic']['genom_mutation_rate'])
        self.max_generation = int(config['Genetic']['max_generation'])
        self.max_fitness = int(config['Genetic']['max_fitness'])
        self.gene_dir = self.util.join_path(
            full_path, config['Genetic']['gene_dir'])
        self.genes_path = self.util.join_path(
            self.gene_dir, config['Genetic']['gene_file'])
        html_checker_dir = self.util.join_path(
            full_path, config['Genetic']['html_checker_dir'])
        self.html_checker = self.util.join_path(
            # html_checker_dir,
            str(""),
            config['Genetic']['html_checker_file'])
        self.html_checker_option = config['Genetic']['html_checker_option']
        self.html_checked_path = self.util.join_path(
            self.html_dir, config['Genetic']['html_checked_file'])
        self.html_eval_place_list = config['Genetic']['html_eval_place'].split(
            '@')
        self.bingo_score = float(config['Genetic']['bingo_score'])
        self.error_score = float(config['Genetic']['error_score'])
        self.result_file = config['Genetic']['result_file']
        self.result_list = []

    # Evaluation.
    def evaluation(self, individual_idx, payload):
        eval_place = "body_tag"
        # Build html syntax.
        html = self.template.render({eval_place: payload})
        eval_html_path = self.util.join_path(
            self.html_dir, self.html_file.replace('*', str(individual_idx)))
        with codecs.open(eval_html_path, 'w', 'utf-8') as fout:
            fout.write(html)

        # Evalute html syntax using tidy.
        enc = locale.getpreferredencoding()
        env_tmp = os.environ.copy()
        env_tmp['PYTHONIOENCODING'] = enc
        subprocess.Popen(
            [self.html_checker, self.html_checker_option, self.html_checked_path, eval_html_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env_tmp)

        # Check html checked result.
        str_eval_result = ''
        with codecs.open(self.html_checked_path, 'r', 'utf-8') as fin:
            str_eval_result = fin.read()
        print(str_eval_result)
        # Check warning and error number.
        # str_pattern = r'.*Tidy found ([0-9]+) warnings and ([0-9]+) errors!*$'
        # str_pattern = r'.*Tidy found ([0-9]+) warning(?:s)? and ([0-9]+) error(?:s)?!*$'
        str_pattern = r'Tidy found (\d+) warning(?:s)? and (\d+) error(?:s)?!'
        # obj_match = re.match(str_pattern, str_eval_result.replace(
        #     '\t', '').replace('\r', '').replace('\n', ''))
        obj_match = re.search(str_pattern, str_eval_result)
        print(obj_match)
        warnings = 0.0
        errors = 0.0
        if obj_match:
            warnings = int(obj_match.group(1)) * -0.1
            errors = int(obj_match.group(2)) * -1.0
        else:
            return None

        # check waf
        detected_count = waf.detect_xss(eval_html_path)
        if detected_count > 0:
            print(f"Found {detected_count} XSS Payloads")
        else:
            print("No XSS Payload found")

        # Compute score.
        int_score = warnings + errors - detected_count

        return int_score

    # Main control.
    def main(self):
        # Load gene list.
        payload = self.payload_list
        idx = self.idx

        # Evaluate
        evaluation_result = self.evaluation(idx, payload)

        return evaluation_result
