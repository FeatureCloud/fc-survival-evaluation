import os
import pickle
import shutil
import threading
import time

import jsonpickle
import pandas as pd
import yaml

from app.algo import check, create_score_df, aggregate_prediction_errors, compute_local_prediction_error, \
    create_cv_accumulation, plot_boxplots


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.y_test_filename = None
        self.y_proba_filename = None

        self.sep = ","
        self.split_mode = None
        self.split_dir = "."
        self.splits = {}
        self.pred_errors = {}
        self.global_errors = {}
        self.score_dfs = {}
        self.cv_averages = None

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_regression_evaluation']
            self.y_test_filename = config['input']['y_true']
            self.y_proba_filename = config['input']['y_pred']
            self.split_mode = config['split']['mode']
            self.split_dir = config['split']['dir']

        if self.split_mode == "directory":
            self.splits = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.split_dir}') if f.is_dir()])
        else:
            self.splits[self.INPUT_DIR] = None

        for split in self.splits.keys():
            os.makedirs(split.replace("/input/", "/output/"), exist_ok=True)
        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_preprocess = 3
        state_aggregate_prediction_errors = 4
        state_wait_for_prediction_errors = 5
        state_compute_scores = 6
        state_writing_results = 7
        state_finishing = 8

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:

            # Local computations

            if state == state_initializing:
                print("[CLIENT] Initializing")
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input
                    print("[CLIENT] Coordinator", self.coordinator)

            if state == state_read_input:
                print('[CLIENT] Read input and config')
                self.read_config()

                for split in self.splits.keys():
                    y_test_path = split + "/" + self.y_test_filename
                    if self.y_test_filename.endswith(".csv"):
                        y_test = pd.read_csv(y_test_path, sep=",")
                    elif self.y_test_filename.endswith(".tsv"):
                        y_test = pd.read_csv(y_test_path, sep="\t")
                    else:
                        y_test = pickle.load(y_test_path)

                    y_pred_path = split + "/" + self.y_proba_filename
                    if self.y_proba_filename.endswith(".csv"):
                        y_proba = pd.read_csv(y_pred_path, sep=",")
                    elif self.y_proba_filename.endswith(".tsv"):
                        y_proba = pd.read_csv(y_pred_path, sep="\t")
                    else:
                        y_proba = pickle.load(y_pred_path)
                    y_test, y_pred = check(y_test, y_proba)
                    self.splits[split] = [y_test, y_pred]
                state = state_preprocess

            if state == state_preprocess:
                for split in self.splits.keys():
                    y_test = self.splits[split][0]
                    y_pred = self.splits[split][1]
                    self.pred_errors[split] = compute_local_prediction_error(y_test, y_pred)

                data_to_send = jsonpickle.encode(self.pred_errors)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregate_prediction_errors
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_prediction_errors
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_prediction_errors:
                print("[CLIENT] Wait for prediction errors")
                self.progress = 'wait for prediction_errors'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregated prediction_errors from coordinator.")
                    self.global_errors = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_compute_scores

            if state == state_compute_scores:
                maes = []
                maxs = []
                rmses = []
                mses = []
                medaes = []

                for split in self.splits.keys():
                    self.score_dfs[split], data = create_score_df(self.global_errors[split])
                    maes.append(data[0])
                    maxs.append(data[1])
                    rmses.append(data[2])
                    mses.append(data[3])
                    medaes.append(data[4])
                if len(self.splits.keys()) > 1:
                    self.cv_averages = create_cv_accumulation(maes, maxs, rmses, mses, medaes)

                state = state_writing_results

            if state == state_writing_results:
                print('[CLIENT] Save results')
                for split in self.splits.keys():
                    self.score_dfs[split].to_csv(split.replace("/input/", "/output/") + "/scores.csv", index=False)

                if len(self.splits.keys()) > 1:
                    self.cv_averages.to_csv(self.OUTPUT_DIR + "/cv_evaluation.csv", index=False)

                    plt = plot_boxplots(self.cv_averages, title=f'{len(self.splits.keys())}-fold Cross Validation' )
                    plt.write_image(f'{self.OUTPUT_DIR}/boxplot.png', format="png", engine="kaleido")
                    plt.write_image(f'{self.OUTPUT_DIR}/boxplot.svg', format="svg", engine="kaleido")
                    plt.write_image(f'{self.OUTPUT_DIR}/boxplot.pdf', format="pdf", engine="kaleido")

                if self.coordinator:
                    self.data_incoming = ['DONE']
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            if state == state_finishing:
                print("Finishing", flush=True)
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            # GLOBAL AGGREGATIONS

            if state == state_aggregate_prediction_errors:
                print("[CLIENT] Aggregate prediction errors")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    for split in self.splits.keys():
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        self.global_errors[split] = aggregate_prediction_errors(split_data)
                    data_to_broadcast = jsonpickle.encode(self.pred_errors)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_compute_scores
                    print(f'[CLIENT] Broadcasting aggregated prediction errors to clients', flush=True)

            time.sleep(1)


logic = AppLogic()
