import os
import sys
import configparser
from util import Utilty
from ga_main import GeneticAlgorithm
from gan_main import GAN
from jinja2 import Environment, FileSystemLoader

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

    # Selenium setting value.
    window_width = int(config['Selenium']['window_width'])
    window_height = int(config['Selenium']['window_height'])
    position_width = int(config['Selenium']['position_width'])
    position_height = int(config['Selenium']['position_height'])

    # Setting template.
    env = Environment(loader=FileSystemLoader(html_dir))
    template = env.get_template(html_template)

    # Create Web driver.
    obj_browser = webdriver.Firefox(
        executable_path="./web_driver/geckodriver")

    # Browser setting.
    obj_browser.set_window_size(window_width, window_height)
    obj_browser.set_window_position(position_width, position_height)

    # Generate many individuals from ga result.
    util.print_message(
        NOTE, 'Generate individual using Generative Adversarial Network.')
    gan = GAN(template, obj_browser)
    gan.main()

    # Close browser.
    obj_browser.close()
