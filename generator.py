import os
import sys
import configparser
from selenium import webdriver
from util import Utilty
from ga_main import GeneticAlgorithm
from jinja2 import Environment, FileSystemLoader
from gan_main import GAN

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

if __name__ == "__main__":
    util = Utilty()

    # Read config.ini
    full_path = os.path.dirname(os.path.abspath(__file__))
    config = configparser.ConfigParser()
    try:
        config.read(util.join_path(full_path, 'config.ini'))
    except FileExistsError as e:
        util.print_message(FAIL, 'File exists error: {}'.format(e))
        sys.exit(1)

    # Common setting value.
    html_dir = util.join_path(full_path, config['Common']['html_dir'])
    html_template = config['Common']['html_template']

    # Genetic Algorithm setting value.
    html_eval_place_list = config['Genetic']['html_eval_place'].split('@')
    max_try_num = int(confg['Genetic']['max_try_num'])

    # Selenium setting value.
    driver_dir = util.join_path(full_path, config['Selenium']['driver_dir'])
    driver_list = config['Selenium']['driver_list'].split('@')
    window_width = int(config['Selenium']['window_width'])
    window_height = int(config['Selenium']['window_height'])
    position_width = int(config['Selenium']['position_width'])
    position_height = int(config['Selenium']['position_height'])

    # Setting template.
    env = Environment(loader=FileSystemLoader(html_dir))
    template = env.get_template(html_template)

    # Start revolution using each brownser.
    for browser in driver_list:
        # Create Web driver.
        obj_browser = webdriver.Firefox(
            executable_path=util.join_path(driver_dir, browser))
        util.print_message(NOTE, 'Launched : {} {}'.format(
            obj_browser.capabilities['browserName'], obj_browser.capabilities['browserVerion']))

        # Browser setting.
        obj_browser.set_window_size(window_width, window_height)
        obj_browser.set_window_position(position_width, position_height)

        # Create a few indiciduals from gene list.
        for idx in range(max_try_num):
            util.print_message(
                NONE, '{}/{} Create individuals using Genetic Algorithm.'.format(idx + 1, max_try_num))
            ga = GeneticAlgorithm
