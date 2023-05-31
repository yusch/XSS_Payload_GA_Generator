import os
import sys
import random
import codecs
import configparser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from util import Utilty

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.

# Generative Adversarial Network class.


class GAN:
    def __init__(self, template, browser):
        self.util = Utilty()
        self.template = template
        self.browser = browser

        # Read config.ini
        full_path = os.path.dirname(os.path.abspath(__file__))
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(full_path, 'config.ini'))
        except FileExistsError as e:
            self.util.print_exception(e, 'File exists error: {}'.format(e))
            sys.exit(1)

        # Common setting value.
        self.wait_time = float(config['Common']['wait_time'])
        self.html_dir = self.util.join_path(
            full_path, config['Common']['html_dir'])
        self.html_file = config['Common']['gan_html_file']
        self.result_dir = self.util.join_path(
            full_path, config['Common']['result_dir'])
        self.eval_html_path = self.util.join_path(
            self.html_dir, self.html_file)

        # Genetic Algorithm setting value.
        self.genom_length = int(config['Genetic']['genom_length'])
        self.gene_dir = self.util.join_path(
            full_path, config['Genetic']['gene_dir'])
        self.genes_path = self.util.join_path(
            self.gene_dir, config['Genetic']['gene_file'])
        self.ga_result_file = config['Genetic']['result_file']
        self.eval_place_list = config['Genetic']['html_eval_place'].split('@')

        # Generative Adversarial Network setting value.
        self.input_size = int(config['GAN']['input_size'])
        self.batch_size = int(config['GAN']['batch_size'])
        self.num_epoch = int(config['GAN']['num_epoch'])
        self.max_sig_num = int(config['GAN']['max_sig_num'])
        self.max_explore_codes_num = int(
            config['GAN']['max_explore_codes_num'])
        self.max_synthetic_num = int(config['GAN']['max_synthetic_num'])
        self.weight_dir = self.util.join_path(
            full_path, config['GAN']['weight_dir'])
        self.gen_weight_file = config['GAN']['generator_weight_file']
        self.dis_weight_file = config['GAN']['discriminator_weight_file']
        self.gan_result_file = config['GAN']['result_file']
        self.gan_vec_result_file = config['GAN']['vec_result_file']
        self.generator = None

        # Load gene list.
        self.df_genes = pd.read_csv(
            self.genes_path, encoding='utf-8').fillna('')
        self.flt_size = len(self.df_genes) / 2.0

        # Path of trained weight.
        self.weight_path = self.util.join_path(self.weight_dir,
                                               self.gen_weight_file.replace('*', str(self.num_epoch - 1)))

    # Build generator model.
    def generator_model(self):
        def forward(x):
            x = F.leaky_relu(F.linear(x, torch.zeros(
                self.input_size, self.input_size*10)), 0.2)
            x = F.dropout(x, 0.5, training=True)
            x = F.leaky_relu(F.linear(x, torch.zeros(
                self.input_size*10, self.input_size*10)), 0.2)
            x = F.dropout(x, 0.5, training=True)
            x = F.leaky_relu(F.linear(x, torch.zeros(
                self.input_size*10, self.input_size*5)), 0.2)
            x = F.dropout(x, 0.5, training=True)
            x = torch.tanh(F.linear(x, torch.zeros(
                self.input_size*5, self.genom_length)))
            return x
        return forward

    # Build discriminator model.
    def discriminator_model(self):
        def forward(x):
            x = F.leaky_relu(F.linear(x, torch.zeros(
                self.genom_length, self.genom_length*10)), 0.2)
            x = F.leaky_relu(F.linear(x, torch.zeros(
                self.genom_length*10, self.genom_length*10)), 0.2)
            x = F.linear(x, torch.zeros(self.genom_length*10, 1))
            x = torch.sigmoid(x)
            return x

        return forward

    # Train GAN model (generate injection codes).
    def train(self, list_sigs):
        # Load train data (=ga result).
        X_train = []
        X_train = np.array(list_sigs)
        X_train = (X_train.astype(np.float32) - self.flt_size) / self.flt_size

        # Build discriminator.
        discriminator = self.discriminator_model()
        d_opt = optim.SGD(discriminator.parameters(), lr=0.1,
                          momentum=0.1, weight_decay=1e-5)
        criterion = nn.BCELoss()

        # Build generator and discriminator (fixed weight of discriminator).
        discriminator.trainable = False
        self.generator = self.generator_model()
        g_opt = optim.SGD(self.generator.parameters(), lr=0.1, momentum=0.3)

        # Execute train.
        num_batches = int(len(X_train) / self.batch_size)
        lst_scripts = []
        for epoch in range(self.num_epoch):
            for batch in range(num_batches):
                # Create noise for inputting to generator.
                noise = torch.tensor(
                    np.random.uniform(-1, 1, size=(self.batch_size, self.input_size)), dtype=torch.float32)

                # Generate new injection code using noise.
                generated_codes = self.generator(noise)

                # Update weight of discriminator.
                image_batch = torch.tensor(
                    X_train[batch * self.batch_size:(batch + 1) * self.batch_size], dtype=torch.float32)
                X = image_batch
                y = torch.tensor([np.random.uniform(0.7, 1.2)
                                 for _ in range(self.batch_size)], dtype=torch.float32)
                d_opt.zero_grad()
                d_output = discriminator(X)
                d_loss_real = criterion(d_output, y)
                d_loss_real.backward()

                X = generated_codes
                y = torch.tensor([np.random.uniform(0.0, 0.3)
                                 for _ in range(self.batch_size)], dtype=torch.float32)
                d_output = discriminator(X.detach())
                d_loss_fake = criterion(d_output, y)
                d_loss_fake.backward()
                d_loss = d_loss_real + d_loss_fake
                d_opt.step()

                # Update weight of generator.
                noise = torch.tensor(
                    np.random.uniform(-1, 1, size=(self.batch_size, self.input_size)), dtype=torch.float32)
                g_opt.zero_grad()
                g_output = discriminator(self.generator(noise))
                g_loss = criterion(g_output, torch.ones(
                    self.batch_size, dtype=torch.float32))
                g_loss.backward()
                g_opt.step()

                # Build HTML syntax from generated codes.
                generated_codes_np = generated_codes.detach().numpy()
                for generated_code in generated_codes_np:
                    lst_genom = []
                    for gene_num in generated_code:
                        gene_num = (gene_num * self.flt_size) + self.flt_size
                        gene_num = int(np.round(gene_num))
                        if gene_num == len(self.df_genes):
                            gene_num -= 1
                        lst_genom.append(int(gene_num))
                    str_html = self.util.transform_gene_num2str(
                        self.df_genes, lst_genom)
                    self.util.print_message(OK, 'Train GAN : epoch={}, batch={}, g_loss={}, d_loss={}, {} ({})'.
                                            format(epoch, batch, g_loss.item(), d_loss.item(),
                                                   np.round(
                                                       (generated_code * self.flt_size) + self.flt_size),
                                                   str_html))

                    # Evaluate generated injection code.
                    for eval_place in self.eval_place_list:
                        # Build html syntax.
                        html = self.template.render({eval_place: str_html})
                        # TODO: Perform evaluation and processing with HTML and Selenium
                        with codecs.open(self.eval_html_path, 'w', 'utf-8') as faout:
                            faout.write(html)

                        # Evaluate individual using selenium.
                        selenium_score, error_flag = self.util.check_individual_selenium(
                            self.obj_browser, self.eval_html_path)
                        if error_flag:
                            continue

                        # Check generated individual using selenium.
                        if selenium_score > 0:
                            self.util.print_message(
                                WARNING, 'Detect running scriptL: "{}" in {}.'.format(str_html, eval_place))
                            # Save running script.
                            lst_scripts.append([eval_place, str_html])

                    # TODO: Perform weight saving of networks each epoch
                    # Save weights of network each epoch.
                    self.generator.save_weights(self.utl.join_path(
                        self.weight_dir, self.gen_weight_file.replace('*', str(epoch))))
                    discriminator.save_weights(self.utl.join_path(
                        self.weight_dir, self.dis_weight_file.replace('*', str(epoch))))

                return lst_scripts

            # Transform from generated codes to gene list.

            def transform_code2gene(self, generated_code):
                lst_genom = []
                generated_code_np = generated_code.detach().numpy()
                for gene_num in generated_code_np:
                    gene_num = (gene_num * self.flt_size) + self.flt_size
                    gene_num = int(np.round(gene_num))
                    if gene_num == len(self.df_genes):
                        gene_num -= 1
                    lst_genom.append(int(gene_num))
                return lst_genom

            # Mean of two vectors.

            def mean_vector(self, vector1, vector2):
                return (vector1 + vector2) / 2

            # Main control.
            def main(self):
                # Define saving path.
                gan_save_path = self.util.join_path(
                    self.result_dir, self.gan_result_file.replace('*', self.obj_browser.name))
                vec_save_path = self.util.join_path(
                    self.result_dir, self.gan_vec_result_file.replace('*', self.obj_browser.name))

                # Start generating injection code.
                if os.path.exists(self.weight_path):
                    # Load trained model.
                    self.generator = self.generator_model()
                    self.generator.load_state_dict(
                        torch.load(self.weight_path))

                    # Explore the valid injection codes.
                    valid_code_list = []
                    result_list = []
                    for idx in range(self.max_explore_codes_num):
                        self.util.print_message(
                            NOTE, '{}/{} Explore valid injection code.'.format(idx + 1, self.max_explore_codes_num))
                        # Generate injection codes.
                        noise = torch.tensor(
                            [np.random.uniform(-1, 1, self.input_size) for _ in range(1)], dtype=torch.float32)
                        generated_codes = self.generator(noise)
                        str_html = self.util.transform_gene_num2str(
                            self.df_genes, self.transform_code2gene(generated_codes[0]))

                        # Evaluate injection code using selenium.
                        for eval_place in self.eval_place_list:
                            html = self.template.render({eval_place: str_html})
                            with codecs.open(self.eval_html_path, 'w', encoding='utf-8') as fout:
                                fout.write(html)

                            selenium_score, error_flag = self.util.check_individual_selenium(
                                self.obj_browser, self.eval_html_path)
                            if error_flag:
                                continue

                            # Check generated injection code.
                            if selenium_score > 0:
                                self.util.print_message(
                                    WARNING, 'Find valid injection code: "{}" in {}.'.format(str_html, eval_place))
                                valid_code_list.append([str_html, noise])
                                result_list.append([eval_place, str_html])

                    # Save generated injection codes.
                    if not os.path.exists(gan_save_path):
                        pd.DataFrame(result_list, columns=['eval_place', 'injection_code']).to_csv(
                            gan_save_path, mode='w', header=True, index=False)
                    else:
                        pd.DataFrame(result_list).to_csv(
                            gan_save_path, mode='a', header=False, index=False)

                    # Synthesize injection codes.
                    vector_result_list = []
                    for idx in range(self.max_synthetic_num):
                        noise_idx1 = np.random.randint(0, len(valid_code_list))
                        noise_idx2 = np.random.randint(0, len(valid_code_list))
                        self.util.print_message(
                            NOTE, '{}/{} Synthesize injection codes.'.format(idx+1, self.max_synthetic_num))
                        self.util.print_message(OK, 'Use two injection codes : ({}) + ({}).'.format(
                            valid_code_list[noise_idx1][0], valid_code_list[noise_idx2][0]))

                        # Generate injection codes.
                        synthesized_noise = self.vector_mean(
                            valid_code_list[noise_idx1][1], valid_code_list[noise_idx2][1])
                        generated_codes = self.generator(synthesized_noise)
                        str_html = self.util.transform_gene_num2str(
                            self.df_genes, self.transform_code2gene(generated_codes[0]))

                        # Evaluate synthesized injection code using selenium.
                        for eval_place in self.eval_place_list:
                            hit_flag = 'Failure'
                            html = self.template.render({eval_place: str_html})
                            with codecs.open(self.eval_html_path, 'w', encoding='utf-8') as fout:
                                fout.write(html)

                            selenium_score, error_flag = self.util.check_individual_selenium(
                                self.obj_browser, self.eval_html_path)
                            if error_flag:
                                continue

                            # Check synthesized injection code using selenium.
                            if selenium_score > 0:
                                self.util.print_message(
                                    WARNING, 'Find running script: "{}".'.format(str_html))
                                hit_flag = 'Bingo'

                            # Save running script.
                            vector_result_list.append(
                                [eval_place, str_html, valid_code_list[noise_idx1][0], valid_code_list[noise_idx2][0], hit_flag])

                    # Save synthesized injection codes.
                    if not os.path.exists(vec_save_path):
                        pd.DataFrame(vector_result_list, columns=['eval_place', 'synthesized_code', 'origin_code1', 'origin_code2', 'bingo']).to_csv(
                            vec_save_path, mode='w', header=True, index=False)
                    else:
                        pd.DataFrame(vector_result_list).to_csv(
                            vec_save_path, mode='a', header=False, index=False)
                else:
                    # Load created individuals by Genetic Algorithm.
                    sig_path = self.util.join_path(
                        self.result_dir, self.ga_result_file.replace('*', self.obj_browser.name))
                    df_temp = pd.read_csv(
                        sig_path, encoding='utf-8').fillna('')
                    df_sigs = df_temp[~df_temp.duplicated()]

                    list_sigs = []
                    # Extract genom list from ga result.
                    for idx in range(len(df_sigs)):
                        list_temp = df_sigs['sig_vector'].values[idx].replace(
                            '[', '').replace(']', '').split(',')
                        list_sigs.append([int(s) for s in list_temp])

                    # Generate individuals (=injection codes).
                    lst_scripts = []
                    target_sig_list = []
                    for target_sig in list_sigs:
                        self.util.print_message(
                            NOTE, 'Start generating injection codes using {}'.format(target_sig))
                        target_sig_list.extend(
                            [target_sig for _ in range(self.max_sig_num)])
                    lst_scripts.extend(self.train(target_sig_list))

                    # Save generated injection codes.
                    if not os.path.exists(gan_save_path):
                        pd.DataFrame(lst_scripts, columns=['eval_place', 'injection_code']).to_csv(
                            gan_save_path, mode='w', header=True, index=False)
                    else:
                        pd.DataFrame(lst_scripts).to_csv(
                            gan_save_path, mode='a', header=False, index=False)

                self.util.print_message(
                    NOTE, 'Done generation of injection codes using Generative Adversarial Networks.')
