import os
import sys
import random
import re
import codecs
import subprocess
import time
import locale
import configparser
import pandas as pd
from decimal import Decimal
from util import Utilty

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

# Container of genoes.


class Gene:
    genom_list = None
    evaluation = None

    def __init__(self, genom_list, evaluation):
        self.genom_list = genom_list
        self.evaluation = evaluation

    def getGenom(self):
        return self.genom_list

    def getEvaluation(self):
        return self.evaluation

    def setGenom(self, genom_list):
        self.genom_list = genom_list

    def setEbaluation(self, evaluation):
        self.evaluation

# Genetic Algorithm.


class GeneticAlgorithm:
    def __init__(self, template, browser):
        self.util = Utilty()
        self.template = template
        self.obj_brwoser = browser

        # Read config.ini.
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(self.util.join_path(full_path, 'config.ini'))
        except FileExistsError as e:
            self.util.print_message(FAIL, 'File exists error: {}'.format(e))
            sys.exit(1)
        # Common setting value.
        self.wait_time = float(config['Common']['wait_time'])
        self.html_dir = self.util.join_path(
            full_path, config['Common']['html_dir'])
        self.html_template = config['Common']['html_template']
        self.html_template_path = self.util.join_path(
            self.html_dir, self.html_template)
        self.html_file = config['Common']['ga_html_file']
        self.result_dir = self.util.join_path(
            full_path, config['Common']['result_dir'])

        # Genetic Algorithm setting value.
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
            html_checker_dir, config['Genetic']['html_checker_file'])
        self.html_checker_option = config['Genetic']['html_checker_option']
        self.html_checked_path = self.util.join_path(
            self.html_dir, config['Genetic']['html_checked_file'])
        self.html_eval_place_list = config['Genetic']['html_eval_place'].split(
            '@')
        self.bingo_score = float(config['Genetic']['bingo_score'])
        self.warning_score = float(config['Genetic']['warning_score'])
        self.error_score = float(config['Genetic']['error_score'])
        self.result_file = config['Genetic']['result_file']
        self.result_list = []  # Common setting value.
        self.wait_time = float(config['Common']['wait_time'])
        self.html_dir = self.util.join_path(
            full_path, config['Common']['html_dir'])
        self.html_template = []

    # Create population.
    def create_genom(self, df_gene):
        lst_gene = []
        for _ in range(self.genom_length):
            lst_gene.append(random.randint(0, len(df_gene.index)-1))
        self.util.print_message(
            OK, 'Created individual : {}.'.format(lst_gene))
        return Gene(lst_gene, 0)

    # Evaluation.
    def evaluation(self, obj_ga, df_gene, eval_place, individual_idx):
        # Build html syntax
        indivisual = self.util.transform_gene_num2str(
            df_gene, obj_ga.genom_list)
        html = self.template.render({eval_place: indivisual})
        eval_html_path = self.util.join_path(
            self.html_dir, self.html_file.replace('*', str(individual_idx)))
        with codecs.open(eval_html_path, 'w', encoding='utf-8') as fout:
            fout.write(html)

        # Evvaluate html syntax using tidy.
        command = self.html_checker + ' ' + self.html_checker_option + \
            ' ' + self.html_checked_path + ' ' + eval_html_path
        enc = locale.getpreferredencoding()
        env_tmp = os.environ.copy()
        env_tmp['PYTHONENCODING'] = enc
        subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env_tmp)

        # Check htmol checked result.
        str_eval_result = ''
        with codecs.open(self.html_checked_path, 'r', encoding='utf-8') as fin:
            str_eval_result = fin.read()
        # Check warning and error number.
        str_pattern = r'.*Tidy found([0-9]+) warnings and ([0-9]+) errors.*$'
        obj_match = re.match(str_pattern, str_eval_result.replace(
            '\t', '').replace('\r', '').replace('\n', ''))
        warnings = 0.0
        errors = 0.0
        if obj_match:
            warnings = int(obj_match.group(1)) * -0.1
            errors = int(obj_match.group(2)) * -1.0
        else:
            return None, 1

        # Compute score.
        int_score = warnings + errors

        # Evaluate running script using selenium.
        selenium_score, error_flag = self.util.check_individual_selenium(
            self.obj_brwoser, eval_html_path)
        if error_flag:
            return None, 1
        
        # Check result of selenium.
        if selenium_score > 0:
            self.util.print_message(OK, 'Detect running script: "{}" in {}.'.format(indivisual, eval_place))

            # compute score for running script.
            int_score += self.bingo_score
            self.result_list.append([eval_place, obj_ga.genom_list, indivisual])

            # Output evaluation results.
            self.util.print_message(OK, 'Evaluation result : Browser={} {}, '
                                    'Individual="{} ({})", '
                                    'Score= {}'.format(self.obj_brwoser.name,
                                                       self.obj_brwoser.capabilities['version'].
                                                       indivisual,
                                                       obj_ga.genom_list,
                                                       str(int_score)))
            return int_score, 0 
        
        # select elite indivisual.
        def select(self, obj_ga, elite):
            