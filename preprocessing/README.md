# Preprocessing Scripts

This folder contains all the preprocessing scripts.
The main script works with a zeek folder that contains a `conn.log.labeled`, an `ssl.log.labeled` and a `x509.log` file.
The `conn.log.labeled` and the `ssl.log.labeled` are split into time windows and stored in directories based on the respective time windows.
Then, the feature extractor script is run in each of these directories and generates a `features.csv` file where possible.
Finally, the different `features.csv` files are combined into one file and stored into output folder.

## Usage
The whole process is initiated by a script:

To generate Normal client features
```
python generate_features.py --raw_dir <input_dir> --processed_dir <output_dir> [--time_window 3600] generate_client_features
```

To generate Malware features 
```
python generate_features.py --raw_dir <input_dir> --processed_dir <output_dir> [--time_window 3600] generate_malware_features
```

where `input_dir` is the main zeek folder, `out_dir` is the output directory where the processed features are saved and `time_window` is the length of the time window in seconds (default 3600).

**Note**: `feature_extractor.py` requires some additional files in order to run at the moment. These files are `alexa_top_1000`, `top_level_domain` and `trusted_root_certificates`. The files can be copied to the directory from where the `generate_features.py` script is run.
