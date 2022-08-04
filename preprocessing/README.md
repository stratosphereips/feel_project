# Preprocessing Scripts

This folder contains all the preprocessing scripts.
The main script works with a zeek folder that contains a `conn.log.labeled`, an `ssl.log.labeled` and a `x509.log` file.
The `conn.log.labeled` and the `ssl.log.labeled` are split into time windows and stored in directories based on the respective time windows.
Then, the feature extractor script is run in each of these directories and generates a `features.csv` file where possible.
Finally the different `features.csv` files are combined into one file and stored into the initial zeek folder.

## Usage
The whole process is initiated by a script: 
```
./generate_features.sh <input_dir> <output_dir> <extractor_dir> <time_window>
```

where `input_dir` is the main zeek folder, `out_dir` is the intermediate directory where the split folders are stored, `extractor_dir` is the directory where the `feature_extractor` python script is stored and `time_window` is the length of the time window in seconds.

**Note**: `feature_extractor.py` requires some additional files in order to run at the moment. These files are `alexa_top_1000`, `top_level_domain` and `trusted_root_certificates`. The files can be copied to the directory from where the `generate_features.sh` script is run.

**Note 2**: At the end of the process the `output_dir` can/should be deleted. The script does not do this automatically to avoid painful mistakes.
Feel free to uncomment the line at your own peril.