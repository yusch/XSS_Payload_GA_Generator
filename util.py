#!/bin/env python
import os
import sys
import string
import random
import configparser
import threading
from datetime import datetime
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoAlertPresentException
from tqdm import tqdm

# Printing colors
OK_BLUE = '\033[94m'      # [*]
NOTE_GREEN = '\033[92m'   # [+]
FAIL_RED = '\033[91m'     # [-]
WARN_YELLOW = '\033[93m'  # [!]
ENDC = '\033[0m'
PRINT_OK = OK_BLUE + '[*]' + ENDC
PRINT_NOTE = NOTE_GREEN + '[+]' + ENDC
PRINT_FAIL = FAIL_RED + '[-]' + ENDC
PRINT_WARN = WARN_YELLOW + '[!]' + ENDC

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

# Utilty class


class Utilty:
    def __init__(self):
        # Read config.ini
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(full_path, 'config.ini'))
        except FileExistsError as err:
            self.print_message(FAIL, 'File exists error: {}'.format(err))
            sys.exit(1)

    # Print metasploit's symbol.
    def print_message(self, type, message):
        if os.name == 'nt':
            if type == NOTE:
                print('[+] ' + message)
            elif type == FAIL:
                print('[-] ' + message)
            elif type == WARNING:
                print('[!] ' + message)
            elif type == NONE:
                print(message)
            else:
                print('[*] ' + message)
        else:
            if type == NOTE:
                tqdm.write(PRINT_NOTE + ' ' + message)
            elif type == FAIL:
                tqdm.write(PRINT_FAIL + ' ' + message)
            elif type == WARNING:
                tqdm.write(PRINT_WARN + ' ' + message)
            elif type == NONE:
                tqdm.write(NOTE_GREEN + message + ENDC)
            else:
                tqdm.write(PRINT_OK + ' ' + message)

    # Print exception messages.
    def print_exception(self, e, message):
        self.print_message(WARNING, 'type:{}'.format(type(e)))
        self.print_message(WARNING, 'args:{}'.format(e.args))
        self.print_message(WARNING, '{}'.format(e))
        self.print_message(WARNING, message)

    # Create random string.
    def get_random_token(self, length):
        chars = string.digits + string.ascii_letters
        return ''.join([random.choice(chars) for _ in range(length)])

    # Get current date.
    def get_current_date(self, indicate_format=None):
        if indicate_format is not None:
            date_format = indicate_format
        else:
            date_format = self.report_date_format
        return datetime.now().strftime(date_format)

    # Transform date from string to object.
    def transform_date_object(self, target_date):
        return datetime.strptime(target_date, self.report_date_format)

    # Transform date from object to string.
    def transform_date_string(self, target_date):
        return target_date.strftime(self.report_date_format)

    # Delete control character.
    def delete_ctrl_char(self, origin_text):
        clean_text = ''
        for char in origin_text:
            ord_num = ord(char)
            # Allow LF,CR,SP and symbol, character and numeric.
            if (ord_num == 10 or ord_num == 13) or (32 <= ord_num <= 126):
                clean_text += chr(ord_num)
        return clean_text

    # Join the paths according to environment.
    def join_path(self, path1, path2):
        return os.path.join(path1, path2)

    # Transform gene from number to string.
    def transform_gene_num2str(self, df_gene, individual_genom_list):
        indivisual = ''
        for gene_num in individual_genom_list:
            indivisual += str(df_gene.loc[gene_num].values[0])
            indivisual = indivisual.replace('%s', ' ').replace('&quot;', '"')\
                .replace('%comma', ',').replace('&apos;', "'")
        return indivisual

     # Check individual using selenium.
    # def check_individual_selenium(self, obj_browser, eval_html_path):
    #     # Evaluate running script using selenium.
    #     int_score = 0
    #     error_flag = False
    #
    #     # Refresh browser for next evaluation.
    #     try:
    #         obj_browser.get(eval_html_path)
    #     except Exception as e:
    #         obj_browser.switch_to_alert.accept()
    #         error_flag = True
    #         return int_score, error_flag
    #
    #     # Judge JavaScript (include event handler).
    #     try:
    #         obj_browser.refresh()
    #         ActionChains(obj_browser).move_by_offset(10, 10).perform()
    #         obj_browser.refresh()
    #     except Exception as e:
    #         # Run script.
    #         alert = wait.until(expected_conditions.alert_is_present())
    #         alert_text = alert.text
    #         print(alert_text)
    #         alert.accept()
    #         int_score = 1
    #
    #     return int_score, error_flag

    def check_individual_selenium(self, obj_browser, eval_html_path):
        # Evaluate running script using selenium.
        int_score = 0
        error_flag = False

        # Refresh browser for next evaluation.
        try:
            obj_browser.execute_script(
                "window.onbeforeunload = function() {};")
            obj_browser.get(
                "file://" + eval_html_path)
        except Exception as e:
            # if obj_browser.switch_to.alert:
            #     obj_browser.switch_to.alert.accept()
            error_flag = True
            return int_score, error_flag

        # Judge JavaScript (include event handler).
        try:
            obj_browser.refresh()
            obj_browser.execute_script("window.scrollTo(10, 10)")
            obj_browser.refresh()
        except Exception as e:
            # Handle multiple alerts.
            while True:
                try:
                    alert = obj_browser.switch_to.alert
                    alert_text = alert.text
                    alert.accept()
                    # Process the alert based on its type.
                    if "prompt" in alert_text:
                        # Handle prompt.
                        alert.send_keys('')
                        alert.accept()
                        pass
                    elif "confirm" in alert_text:
                        # Handle confirm.
                        alert.dismiss()
                        pass
                    elif "alert" in alert_text:
                        # Handle alert.
                        alert.accept()
                        pass
                except NoAlertPresentException:
                    break

            int_score = 1

        return int_score, error_flag
