from datetime import datetime
import pytz
from .ExtractFeatures import ExtractFeatures
from .PrintingManager import PrintingManager
import pickle
#from xgboost.sklearn import XGBClassifier
#from xgboost.sklearn import Booster
import numpy as np
from sklearn.preprocessing import LabelEncoder


class ComputeFeatures(ExtractFeatures, PrintingManager):
    def __init__(self):
        super(ComputeFeatures, self).__init__()
        self.file_time_name = str(datetime.strftime(datetime.now(pytz.utc), "%Y-%m-%d_%H-%M"))
        self.data_model = None
        self.tuples_keys = None
        self.result_dict = {}
        self.malware = 0
        self.normal = 0


    def add_cert_to_non_cert_conn(self):
        for key in self.connection_4_tuples.keys():
            """
            implementig feature: connection which have no certificate, but have at least one SNI,
            look, if in certificate_objects_dict is such servername with certificate
            """
            break_v = 0
            if self.connection_4_tuples[key].get_amount_diff_certificates() == 0:

                server_names = self.connection_4_tuples[key].get_SNI_list()
                if len(server_names) != 0:
                    for cert_serial in self.certificate_dict.keys():
                        for server_name in server_names:
                            x509_line = self.certificate_dict[cert_serial].contain_server_name(server_name)
                            if x509_line != 0:
                                self.connection_4_tuples[key].add_ssl_log_2(x509_line)
                                # print("This Certificate was added after process:", "cert_serial:", cert_serial, "server_name=",server_name, "4-tuple=", key, "label:", self.connection_4_tuples[key].get_label_of_connection())
                                break_v = 1
                                break
                        if break_v == 1:
                            break

    def get_numbers_of_certificates(self):
        malware_cert = 0
        normal_cert = 0
        all_cert = len(self.certificate_dict.keys())
        for key in self.certificate_dict.keys():
            if self.certificate_dict[key].is_malware_cert():
                malware_cert += 1
            else:
                normal_cert += 1
        print("All certificates:", all_cert)
        print("Malware certificates:", malware_cert)
        print("Normal certificates:", normal_cert)

    def normalize_data(self, data):
        # These values are max values from learning data from normalization. We have to normalize these new data
        # by this values.
        maxs = [221044.0, 2052217.81665, 2058176.55791, 1.0, 1.33529714507e+11, 1.44846580408e+11, 11668903.3436,
                1.0, 32749474.0, 22193611.0, 36718818.0321, 18361450.8651, 5268.0, 180224.0, 1.0, 22000.0, 3492.5,
                956.0, 176.0, 883.0, 22.8304763079, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 65537.0, 0.0, 0.0, 0.0]
        for i in range(0, len(data[0])):
            # max = 0
            # for j in range(len(data)):
            #     if max < data[j][i]:
            #         max = data[j][i]
            if maxs[i] != 0:
                for j in range(len(data)):
                    if data[j][i] != -1:
                        data[j][i] = data[j][i] / float(maxs[i])
        return data

    def prepare_data(self):
        # data model is lines of ssl connect unit.
        self.data_model = []
        self.tuples_keys = []
        for key in self.connection_4_tuples.keys():
            line = []
            self.tuples_keys.append(key)

            line.append(self.connection_4_tuples[key].get_number_of_flows())
            line.append(self.connection_4_tuples[key].get_average_of_duration())
            line.append(self.connection_4_tuples[key].get_standard_deviation_duration())
            line.append(self.connection_4_tuples[key].get_percent_of_standard_deviation_duration())
            line.append(self.connection_4_tuples[key].get_total_size_of_flows_orig())
            line.append(self.connection_4_tuples[key].get_total_size_of_flows_resp())
            line.append(self.connection_4_tuples[key].get_ratio_of_sizes())
            line.append(self.connection_4_tuples[key].get_percent_of_established_states())
            line.append(self.connection_4_tuples[key].get_inbound_pckts())
            line.append(self.connection_4_tuples[key].get_outbound_pckts())
            line.append(self.connection_4_tuples[key].get_periodicity_average())
            line.append(self.connection_4_tuples[key].get_periodicity_standart_deviation())
            line.append(self.connection_4_tuples[key].get_ssl_ratio())
            line.append(self.connection_4_tuples[key].get_average_public_key())
            line.append(self.connection_4_tuples[key].get_tls_version_ratio())
            line.append(self.connection_4_tuples[key].get_average_of_certificate_length())
            line.append(self.connection_4_tuples[key].get_standart_deviation_cert_length())
            line.append(self.connection_4_tuples[key].is_valid_certificate_during_capture())
            line.append(self.connection_4_tuples[key].get_amount_diff_certificates())
            line.append(self.connection_4_tuples[key].get_number_of_domains_in_certificate())
            line.append(self.connection_4_tuples[key].get_certificate_ratio())
            line.append(self.connection_4_tuples[key].get_number_of_certificate_path())
            line.append(self.connection_4_tuples[key].x509_ssl_ratio())
            line.append(self.connection_4_tuples[key].SNI_ssl_ratio())
            line.append(self.connection_4_tuples[key].self_signed_ratio())
            line.append(self.connection_4_tuples[key].is_SNIs_in_SNA_dns())
            line.append(self.connection_4_tuples[key].get_SNI_equal_DstIP())
            line.append(self.connection_4_tuples[key].is_CNs_in_SNA_dns())
            # New features
            line.append(self.connection_4_tuples[key].ratio_of_differ_SNI_in_ssl_log())
            line.append(self.connection_4_tuples[key].ratio_of_differ_subject_in_ssl_log())
            line.append(self.connection_4_tuples[key].ratio_of_differ_issuer_in_ssl_log())
            line.append(self.connection_4_tuples[key].ratio_of_differ_subject_in_cert())
            line.append(self.connection_4_tuples[key].ratio_of_differ_issuer_in_cert())
            line.append(self.connection_4_tuples[key].ratio_of_differ_sandns_in_cert())
            line.append(self.connection_4_tuples[key].ratio_of_same_subjects())
            line.append(self.connection_4_tuples[key].ratio_of_same_issuer())
            line.append(self.connection_4_tuples[key].ratio_is_same_CN_and_SNI())
            line.append(self.connection_4_tuples[key].average_certificate_exponent())
            line.append((self.connection_4_tuples[key].is_SNI_in_top_level_domain()))
            line.append(self.connection_4_tuples[key].ratio_certificate_path_error())
            line.append(self.connection_4_tuples[key].ratio_missing_cert_in_cert_path())
            # line.append(self.connection_4_tuples[key].ratio_of_root_certificates())
            self.data_model.append(line)

        # normalize data
        # self.data_model = self.normalize_data(self.data_model)

    def detect(self):
        clf_xgboost = XGBClassifier()
        booster = Booster()
        try:
            booster.load_model('./xgboost_2017_09_22.bin')
        except IOError:
            print('Error: No ML module to read.')
        clf_xgboost._Booster = booster
        # clf_xgboost._le = LabelEncoder().fit(['Malware', 'Normal'])
        clf_xgboost._le = LabelEncoder().fit([1, 0])
        results = clf_xgboost.predict(np.array(self.data_model))


        for i in range(len(self.tuples_keys)):
            label = ''
            if results[i] == 0:
                label = 'Normal'
                self.normal += 1
            else:
                label = 'Suspicious'
                self.malware += 1
            self.result_dict[self.tuples_keys[i]] = label
