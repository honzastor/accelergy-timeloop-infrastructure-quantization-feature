* The `construct_workload` folder contains scripts for automatic creation of problem conv layers into Timeloop's workload yaml format
    - `construct_workloads.py` script is used for generating your own workloads in Timeloop format
    - first you can look into the `temps/cnn_layers.yaml` file to see example descriptions of model's conv layers
    - than you can use the `parse_model.py` script to load in Keras or PyTorch model and parse it into the list of workload layers used by the `construct_workloads.py`
    - examle run may be: `python3 parse_model.py --api_name keras --model mobilenet -o keras_mobilenet`
    - this will create the `keras_mobilenet.yaml` file containing the model's conv layers shapes into the `parsed_models` folder
    - then proceed to run the main constructor script `python3 construct_workloads.py <my_dnn_model_name>.yaml`
    - for additional scripts arguments examine their help messages

* The `parse_oaves` folder contains script `oaves_process_data.py` which sorts and cleans an input `.oaves.csv`, removing the sub-optimal mappings

* The `parse_xml` folder contains script `parse_xml_timeloop_output.py` for parsing the generated xml stats into .pkl format for further analysis

* The `plot_graphs` contains jupyter notebook `plot_graphs.ipynb` to plot graphs for the data contained in the `../scenarios` folder