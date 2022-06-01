import logging
import os
import shutil
import threading
import time
from collections import defaultdict
from typing import Tuple, Any, Dict, List, Optional

import jsonpickle
import numpy as np
import pandas as pd
import yaml
from nptyping import NDArray, Bool

from app.algo import LocalConcordanceIndex, calculate_cindex_on_local_data, GlobalConcordanceIndexEvaluations, \
    AggregatedConcordanceIndex

MINIMUM_CONCORDANT_PAIRS = 3


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

        # === Privacy ===
        self.min_concordant_pairs = None

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
        self.y_pred_filename = None

        self.objective = None

        self.sep = ","
        self.mode = None
        self.dir = "."

        self.label_time_to_event = None
        self.label_event = None
        self.label_pred_time_to_event = None
        self.event_truth_value = None

        self.actual: Dict[str, NDArray] = {}
        self.predicted: Dict[str, NDArray] = {}

        self.local_evaluations: Dict[str, LocalConcordanceIndex] = {}
        self.global_results: Dict[str, AggregatedConcordanceIndex] = {}

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data_read):
        # This method is called when new data arrives
        logging.debug("Process incoming data....")
        data_read = data_read.read()
        logging.debug(f"Add data to incoming: {data_read}")
        self.data_incoming.append(data_read)

    def handle_outgoing(self):
        logging.debug("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_survival_evaluation']
            self.min_concordant_pairs = max(config['privacy']['min_concordant_pairs'], MINIMUM_CONCORDANT_PAIRS)

            self.y_test_filename = config['input']['y_test']
            self.y_pred_filename = config['input']['y_pred']

            self.sep = config['format']['sep']
            self.label_time_to_event = config["format"]["label_survival_time"]
            self.label_event = config["format"]["label_event"]
            self.label_pred_time_to_event = config["format"]["label_predicted_time"]
            self.event_truth_value = config["format"].get("event_truth_value", True)  # default value

            self.objective = config["parameters"]["objective"]
            possible_objectives = ['ranking', 'regression']
            if self.objective not in possible_objectives:
                raise ValueError(f"Unknown objective. Choose one of {', '.join(possible_objectives)}")

            self.mode = config['split']['mode']
            self.dir = config['split']['dir']

        if self.mode == "directory":
            self.actual = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.dir}') if f.is_dir()])
            self.predicted = dict.fromkeys(self.actual)
        else:
            self.actual[self.INPUT_DIR] = {}
            self.predicted[self.INPUT_DIR] = {}

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        for split in self.actual.keys():
            os.makedirs(split.replace("/input/", "/output/"), exist_ok=True)
        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)

    @staticmethod
    def get_column(dataframe: pd.DataFrame, col_name: str) -> pd.Series:
        try:
            return dataframe[col_name]
        except KeyError as e:
            logging.error(f"Column {col_name} does not exist in the data")
            raise e

    @staticmethod
    def event_value_to_truth_array(event: NDArray[Any], truth_value: Any) -> NDArray[Bool]:
        if truth_value is True and event.dtype == np.dtype('bool'):  # nothing to do...
            return event

        truth_array = (event == truth_value)
        return truth_array

    def read_data_frame(self, path):
        logging.info(f"Read data file at {path}")
        dataframe = pd.read_csv(path, sep=self.sep)
        logging.debug(f"Dataframe:\n{dataframe}")
        return dataframe

    def read_survival_data(self, path) -> Tuple[pd.DataFrame, NDArray]:
        X: pd.DataFrame = self.read_data_frame(path)

        event = self.get_column(X, self.label_event)
        logging.debug(f"event:\n{event}")
        event_occurred = self.event_value_to_truth_array(event.to_numpy(), self.event_truth_value)
        logging.debug(f"event_occurred:\n{event_occurred}")

        time_to_event = self.get_column(X, self.label_time_to_event)
        logging.debug(f"time_to_event:\n{time_to_event}")

        X.drop([self.label_event, self.label_time_to_event], axis=1, inplace=True)
        logging.debug(f"features:\n{X}")
        y = np.zeros(X.shape[0], dtype=[('Status', '?'), ('Survival', '<f8')])  # TODO
        y['Status'] = event
        y['Survival'] = time_to_event

        return [X, y]

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_local_evaluation = 3
        state_aggregation_of_evaluation = 4
        state_waiting_for_evaluation = 5
        state_writing_results = 6
        state_shutdown = 7

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

                for split in self.actual.keys():
                    _, y = self.read_survival_data(os.path.join(split, self.y_test_filename))
                    self.actual[split] = y

                    predictions_csv_df: pd.DataFrame = self.read_data_frame(os.path.join(split, self.y_pred_filename))
                    self.predicted[split] = predictions_csv_df[self.label_pred_time_to_event].to_numpy()
                    logging.debug(f"Predictions: {self.predicted[split]}")
                state = state_local_evaluation

            if state == state_local_evaluation:
                self.progress = f'local evaluation...'
                self.local_evaluations: Dict[str, LocalConcordanceIndex] = dict.fromkeys(self.actual)
                self.public_local_evaluations: Dict[str, Optional[LocalConcordanceIndex]] = dict.fromkeys(self.actual)

                # calculate c-index for each split
                for split in self.actual.keys():
                    event_indicator = self.actual[split]['Status']
                    event_time = self.actual[split]['Survival']

                    estimate = self.predicted[split]
                    if self.objective == 'regression':
                        estimate = -estimate

                    local_result = calculate_cindex_on_local_data(event_indicator, event_time, estimate)
                    self.local_evaluations[split] = local_result
                    if local_result.num_concordant_pairs < self.min_concordant_pairs:
                        logging.debug(f'Opt-out')
                        self.public_local_evaluations[split] = None
                    else:
                        self.public_local_evaluations[split] = local_result

                logging.debug(self.local_evaluations)

                # calculate c-index over all splits
                logging.debug('Calculate c-index over all splits')
                actual = np.concatenate([self.actual[split] for split in self.actual.keys()])
                predicted = np.concatenate([self.predicted[split] for split in self.predicted.keys()])

                if self.objective == 'regression':
                    predicted = -predicted

                overall_cindex = calculate_cindex_on_local_data(event_indicator=actual['Status'], event_time=actual['Survival'], estimate=predicted)
                if overall_cindex.num_concordant_pairs < self.min_concordant_pairs:
                    logging.debug(f'Opt-out')
                    overall_cindex = None

                self.local_evaluations[self.INPUT_DIR] = overall_cindex
                self.public_local_evaluations[self.INPUT_DIR] = overall_cindex

                # sending data
                data_to_send = jsonpickle.encode(self.public_local_evaluations)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregation_of_evaluation
                    logging.debug(f'[CONTROLLER] Adding EVALUATION data locally')
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_waiting_for_evaluation
                    logging.debug(f'[CLIENT] Sending EVALUATION data to coordinator')

            if state == state_aggregation_of_evaluation:
                self.progress = f'waiting for evaluation results. {len(self.data_incoming)} of {len(self.clients)}...'
                logging.debug(self.progress)
                if len(self.data_incoming) == len(self.clients):
                    self.progress = f'aggregate local evaluations...'
                    logging.debug("Received data attributes of all clients")
                    results: List[Dict[str, Optional[LocalConcordanceIndex]]] = [jsonpickle.decode(client_data) for
                                                                                 client_data in self.data_incoming]
                    self.data_incoming = []
                    logging.debug(results)

                    # unwrap
                    local_results: Dict[str, List[LocalConcordanceIndex]] = defaultdict(list)
                    for res_dict in results:
                        for split, evaluation in res_dict.items():
                            if evaluation is not None:
                                local_results[split].append(evaluation)

                    self.global_results = dict.fromkeys(local_results)
                    for split, evaluations in local_results.items():
                        aggregator = GlobalConcordanceIndexEvaluations(evaluations)
                        self.global_results[split] = aggregator.calc()

                    data_to_broadcast = jsonpickle.encode(self.global_results)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_writing_results
                    print(f'[CLIENT] Broadcasting EVALUATION data to clients', flush=True)

            if state == state_waiting_for_evaluation:
                self.progress = 'wait for aggregation...'
                logging.debug(self.progress)
                if len(self.data_incoming) > 0:
                    logging.info("[CLIENT] Received EVALUATION aggregation data from coordinator.")
                    self.global_results = jsonpickle.decode(self.data_incoming[0])
                    state = state_writing_results

            if state == state_writing_results:
                self.progress = 'writing results...'
                logging.debug(self.global_results)

                for split, evaluation in self.global_results.items():
                    output_path = os.path.join(split.replace("/input", "/output"), "scores.tab")
                    logging.debug(f"Write output for {split} to {output_path}")
                    with open(output_path, "w") as fh:
                        local: LocalConcordanceIndex = self.local_evaluations[split]
                        fh.write(f"number of samples (local)\t{local.num_samples}\n")
                        fh.write(f"c-index on local data\t{local.cindex}\n")
                        fh.write(f"concordant pairs on local data\t{local.num_concordant_pairs}\n")
                        public_local: Optional[LocalConcordanceIndex] = self.public_local_evaluations[split]
                        fh.write(f"opt-out\t{public_local is None}\n")
                        aggregated: Optional[AggregatedConcordanceIndex] = evaluation
                        if aggregated is not None:
                            fh.write(f"mean c-index\t{aggregated.mean_cindex}\n")
                            fh.write(f"c-index weighted by number of samples\t{aggregated.weighted_cindex_samples}\n")
                            fh.write(f"c-index weighted by number of concordant pairs\t{aggregated.weighted_cindex_concordant_pairs}\n")

                state = state_shutdown

            if state == state_shutdown:
                self.progress = "finished"
                logging.info("Finished")
                self.status_finished = True
                break

            time.sleep(1)


logic = AppLogic()
