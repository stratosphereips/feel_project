from termcolor import colored, cprint


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class PrintingManager(object):
    def __init__(self):
        super(PrintingManager, self).__init__()
        self.verbosity = 1

    def init_hello(self, verbosity):
        self.verbosity = verbosity
        if not verbosity:
            return
        self.space_1 = "    "
        self.space_2 = self.space_1 + self.space_1

        print(
            " _    _ _______ _______ _____   _____     _____  ______ _______ ______ _____ _______ ____  _____     "
        )
        print(
            "| |  | |__   __|__   __|  __ \ / ____|   |  __ \|  ____|__   __|  ____/ ____|__   __/ __ \|  __ \    "
        )
        print(
            "| |__| |  | |     | |  | |__) | (___     | |  | | |__     | |  | |__ | |       | | | |  | | |__) |   "
        )
        print(
            "|  __  |  | |     | |  |  ___/ \___ \    | |  | |  __|    | |  |  __|| |       | | | |  | |  _  /    "
        )
        print(
            "| |  | |  | |     | |  | |     ____) |   | |__| | |____   | |  | |___| |____   | | | |__| | | \ \    "
        )
        print(
            "|_|  |_|  |_|     |_|  |_|    |_____/    |_____/|______|  |_|  |______\_____|  |_|  \____/|_|  \_\   "
        )

        print(
            "\nHTTPS Detector tool. Author: Frantisek Strasak, strasfra@fel.cvut.cz, "
            "verbosity: {}\n".format(self.verbosity)
        )

    def print_data_statistic(self, exit_code):
        if exit_code >= 0:
            if self.verbosity > 0:
                print("Loaded bro files:")
                print(self.space_1 + "conn.log files: {}".format(len(self.conn_files)))
                if self.verbosity > 1:
                    for cfile in self.conn_files:
                        print(self.space_2 + cfile)
                print(self.space_1 + "ssl.log files: {}".format(len(self.ssl_files)))
                if self.verbosity > 1:
                    for sfile in self.ssl_files:
                        print(self.space_2 + sfile)
                print(self.space_1 + "x509.log files: {}".format(len(self.x509_files)))
                if self.verbosity > 1:
                    for xfile in self.x509_files:
                        print(self.space_2 + xfile)
        else:
            if exit_code == -1:
                print("There is no conn.log files.")
            elif exit_code == -2:
                pass
                # print('Ther is no ssl.log files. Maybe this capture does not contain HTTPS connection.')
            elif exit_code == -3:
                print(
                    "There is no x509.log files. Maybe this capture does not contain HTTPS connection."
                )

    def print_detection_result(self):
        print("Number of connection: {}".format(len(self.connection_4_tuples.keys())))
        print(
            colored(
                "Malware: {}".format(self.malware), "red", attrs=["reverse", "blink"]
            )
        )
        print(
            colored(
                "Normal: {}".format(self.normal), "green", attrs=["reverse", "blink"]
            )
        )
        print("")
        print(
            "-----------------------------------------------------------------------------------"
        )
        print(
            "{:12s} {:20s} {:20s} {:10s} {:10s} {:10s}".format(
                "#", "SrcIP", "DstIP", "DstPort", "Protocol", "Label"
            )
        )
        print(
            "-----------------------------------------------------------------------------------"
        )
        for i, key in enumerate(sorted(self.result_dict.keys())):
            s = "{:12s} {:20s} {:20s} {:10s} {:10s} {:10s}".format(
                str(i + 1), key[0], key[1], key[2], key[3], self.result_dict[key]
            )
            if self.result_dict[key] != "Normal":
                print(colored(s, "red", attrs=["reverse", "blink"]))
            else:
                print(colored(s, "green", attrs=["reverse", "blink"]))
