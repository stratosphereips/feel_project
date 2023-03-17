#!/usr/bin/env python3
# Author: Maria Rigaki based on code from Frantisek Strasak strasfra[ampersat]fel.cvut.cz

__version__ = 0.1

import argparse

from time import time
from zeek_source.ComputeFeatures import ComputeFeatures
from zeek_source.PrintingManager import PrintingManager
from zeek_source.ExtractFeatures import ExtractFeatures
import numpy as np
import pandas as pd
import os

CNAMES = [
    "num_flows",
    "avg_dur",
    "std_dev_dur",
    "percent_stdev_dur",
    "total_size_of_flows_orig",
    "total_size_of_flows_resp",
    "ratio_of_sizes",
    "percent_of_established_states",
    "inbound_pckts",
    "outbound_pckts",
    "periodicity_avg",
    "periodicity_stdev",
    "ssl_ratio",
    "average_public_key",
    "tls_version_ratio",
    "avg_cert_length",
    "stdev_cert_length",
    "is_valid_certificate_during_capture",
    "amount_diff_certificates",
    "num_domains_in_cert",
    "cert_ratio",
    "num_certificate_path",
    "x509_ssl_ratio",
    "SNI_ssl_ratio",
    "self_signed_ratio",
    "is_SNIs_in_SNA_dns",
    "SNI_equal_DstIP",
    "is_CNs_in_SNA_dns",
    "ratio_of_differ_SNI_in_ssl_log",
    "ratio_of_differ_subject_in_ssl_log",
    "ratio_of_differ_issuer_in_ssl_log",
    "ratio_of_differ_subject_in_cert",
    "ratio_of_differ_issuer_in_cert",
    "ratio_of_differ_sandns_in_cert",
    "ratio_of_same_subjects",
    "ratio_of_same_issuer",
    "ratio_is_same_CN_and_SNI",
    "avg_certificate_exponent",
    "is_SNI_in_top_level_domain",
    "ratio_certificate_path_error",
    "ratio_missing_cert_in_cert_path",
    "label",
    "detailedlabel",
    "id.orig_h",
    "id.resp_h",
    "id.resp_p",
    "proto",
]


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

    return np.array(extract_features.data_model)


def check_arg():
    # version
    # help
    # read one bro folder
    # read multi folders of bro
    # read suricata eye.json
    # read more suricata eye.jsons

    # Parse the parameters
    parser = argparse.ArgumentParser(
        description="Program HTTPS Detector tool version {}. Author: "
        "Frantisek Strasak, strasfra@fel.cvut.cz. Sebastian Garcia".format(__version__),
        usage="%(prog)s -n <screen_name> [options]",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="0-no verbosity, 1-middle verbosity, 2-high verbosity (default is 1)",
        action="store",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "-V",
        "--version",
        help="{}".format(__version__),
        action="version",
        version="{}".format(__version__),
    )

    parser.add_argument(
        "-z",
        "--zeekfolder",
        help="Path to a folder where all log files are.",
        action="store",
        required=False,
    )
    parser.add_argument(
        "-Z",
        "--zeekfolders",
        help="Multiple captures. Path to a folder where Bro folders are.",
        action="store",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default="features.csv",
        help="Name of the ouput file",
        action="store",
    )

    args = parser.parse_args()

    # Check arguments

    if not (args.zeekfolder or args.zeekfolders):
        parser.error("No action requested, see --help")

    return args


if __name__ == "__main__":
    args = check_arg()

    if args.zeekfolder:
        # print('Bro folder: {}'.format(args.brofolder))
        data = read_one_capture(args.zeekfolder, args.verbose)
    elif args.zeekfolders:
        # print('Bro folders: {}'.format(args.brofolder))
        print("It is not implemeted.")

    if data is None or not data.size:
        data = np.empty(shape=(0, len(CNAMES)))
    df = pd.DataFrame(data, columns=CNAMES)
    # print(df.head())
    df.to_csv(os.path.join(args.zeekfolder, args.output), index=False)
