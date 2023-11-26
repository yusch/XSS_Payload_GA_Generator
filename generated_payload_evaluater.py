import os
import sys
import datetime
import configparser
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from util import Utilty
from generated_payload_evaluater_main import GeneticAlgorithm
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
import setup_log

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

    # Selenium setting value.
    window_width = int(config['Selenium']['window_width'])
    window_height = int(config['Selenium']['window_height'])
    position_width = int(config['Selenium']['position_width'])
    position_height = int(config['Selenium']['position_height'])

    # Setting template.
    env = Environment(loader=FileSystemLoader(html_dir))
    template = env.get_template(html_template)

    # Headless mode option
    options = Options()
    options.add_argument('--headless')

    # Create Web driver.
    obj_browser = webdriver.Firefox(
        executable_path="./web_driver/geckodriver", options=options)

    # Browser setting.
    obj_browser.set_window_size(window_width, window_height)
    obj_browser.set_window_position(position_width, position_height)

    # output print function's contetnts to terminal and log file
    sys.stdout = setup_log.Logger()

    # load payload list
    f = open('payload_list.txt', 'r')
    g = open('evaluate_result.txt', 'w')
    payload_list = f.readlines()

    # Create a few individuals from gene list.
    for idx in tqdm(range(len(payload_list)), desc='main loop'):
        util.print_message(
            NONE, '{}/{} payload evaluated'.format(idx + 1, len(payload_list)))
        ga = GeneticAlgorithm(template, obj_browser, idx, payload_list[idx])
        score = ga.main()
        g.write(f'{payload_list[idx], score}\n')

    # Close browser.
    obj_browser.close()

    # close payload list
    f.close()