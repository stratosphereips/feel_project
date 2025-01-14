# SPDX-FileCopyrightText: 2021 Sebastian Garcia <sebastian.garcia@agents.fel.cvut.cz>
#  SPDX-License-Identifier: GPL-2.0-only
# Author: Frantisek Strasak strasfra[ampersat]fel.cvut.cz
"""
Read bro files.
"""
from time import time
from .ComputeFeatures import ComputeFeatures
from .PrintingManager import PrintingManager
from .ExtractFeatures import ExtractFeatures
import numpy as np


def read_one_capture(path_to_bro_folder, verbosity):
    t0 = time()
    extract_features = ComputeFeatures()
    # Init hello
    extract_features.init_hello(verbosity)
    # Read Bro data.
    exit_code = extract_features.extraction_manager(path_to_bro_folder + "/")
    # Check if we have needed files.
    if exit_code < 0:
        extract_features.print_data_statistic(exit_code)
        return
    # Print data statistic
    extract_features.print_data_statistic(exit_code)
    # Add certificate to connections that does not contain any certificate.
    extract_features.add_cert_to_non_cert_conn()
    # Compute features and save them.
    extract_features.prepare_data()
    print(np.array(extract_features.data_model).shape)
    # extract_features.detect()
    # extract_features.print_detection_result()
    print(
        "<<< All dataset successfuly finished in approximate time: %f"
        % ((time() - t0) / 60.0)
        + " min."
    )
