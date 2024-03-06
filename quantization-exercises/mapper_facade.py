# Author: Jan Klhufek (iklhufek@fit.vut.cz)

import os
import re
import sys
import subprocess
import time
from datetime import datetime
import glob
import json
import gzip
import csv
import shutil
import threading
import yaml
import math
import numpy as np
import multiprocessing
import xml.etree.ElementTree as ET
from timeloop_utils.construct_workloads.create_model_torch import create_pytorch_model
from timeloop_utils.construct_workloads.create_model_keras import create_keras_model
from timeloop_utils.construct_workloads.parse_model_torch import parse_pytorch_model
from timeloop_utils.construct_workloads.parse_model_keras import parse_keras_model
from timeloop_utils.construct_workloads.construct_workloads import json_file_to_dict, construct_workloads
from typing import Tuple, Dict, Optional, Any, Union


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Union[int, list, Any]:
        """
        Custom JSON encoder for handling NumPy data types.

        Args:
            obj (Any): The object to encode into JSON.

        Returns:
            Union[int, list, Any]: Encoded object compatible with JSON.
        """
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)


def dict_to_json(dictionary: Dict, filename: str) -> None:
    """
    Writes a dictionary to a JSON file.

    Args:
        dictionary (Dict): The dictionary to be written to the file.
        filename (str): The name of the file to write the dictionary to.
    """
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file, indent=2, cls=JSONEncoder)


def get_stat(stats: ET.Element, stat: str, cast: type) -> np.ndarray:
    """
    Extracts statistics from XML data.
    Modified code originating from: https://github.com/NVlabs/timeloop/blob/master/scripts/parse_timeloop_output.py

    Args:
        stats (ET.Element): XML element containing the statistics.
        stat (str): The name of the statistic to extract.
        cast (type): The data type to cast the extracted values to.

    Returns:
        np.ndarray: An array of extracted values cast to the specified type.
    """
    items = stats.findall(stat)[0].findall('PerDataSpace')[0].findall('item')
    count = len(items)
    out = np.array([0]*count, dtype=cast)
    for j in range(count):
        if stat == 'ingresses':
            value = sum([cast(i.text) for i in items[j].findall('item')])
        else:
            value = cast(items[j].text)
        out[j] = value
    return out


def extract_scalar_accesses_data(data: str, key: str) -> Optional[Dict[str, float]]:
    """
    Extracts scalar access data from a string based on a given key.

    Args:
        data (str): The string containing the data.
        key (str): The key to look for in the data.

    Returns:
        Optional[Dict[str, float]]: A dictionary containing the extracted data if the key is found, None otherwise.
    """
    regex_part = r"\s*Total scalar accesses\s*:\s*([\d]+)\s*Op per Byte\s*:\s*([\d.]+)"
    pattern = f"=== {key} ==={regex_part}"
    match = re.search(pattern, data)
    if match:
        return {"Total scalar accesses": int(match.group(1)), "Op per Byte": float(match.group(2))}
    return None


def extract_memory_stats(architecture: str, workload: str, data: str, result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts memory statistics from a given data string for a specified architecture.

    Args:
        architecture (str): The name of the architecture (e.g., 'eyeriss', 'simba').
        workload (str): Relative path to the workload configuration file.
        data (str): The string containing the data.
        result_dict (Dict[str, Any]): The dictionary to store the extracted results.

    Returns:
        Dict[str, Any]: A dictionary with the extracted memory statistics.
    """
    # Extract "Word bits"
    word_bits_pattern = r"Word bits\s*:\s*(\d+)"
    word_bits_match = re.search(word_bits_pattern, data)
    if word_bits_match:
        word_bits = int(word_bits_match.group(1))
        result_dict["Word bits"] = word_bits

    # Calculate weights_memory_size
    with open(workload, 'r') as work_file:
        work_data = yaml.safe_load(work_file)

    # Retrieve the values
    C = work_data['problem']['instance']['C']  # input channels
    M = work_data['problem']['instance']['M']  # output channels
    R = work_data['problem']['instance']['R']  # kernel width
    S = work_data['problem']['instance']['S']  # kernel height
    # Check if 'bitwidths' key exists and has a 'Weights' subkey
    if 'bitwidths' in work_data['problem']['instance'] and 'Weights' in work_data['problem']['instance']['bitwidths']:
        Weights_bitwidth = work_data['problem']['instance']['bitwidths']['Weights']
    else:
        Weights_bitwidth = word_bits
    weights = C * M * R * S
    weights_memory_size = weights * Weights_bitwidth

    # Add weights_memory_size to the cache
    result_dict["Weights model memory size [bits]"] = weights_memory_size
    result_dict["Weights model memory size [Bytes]"] = math.ceil(weights_memory_size/8)
    result_dict["Weights model memory size [Words]"] = math.ceil(weights/math.floor(word_bits/Weights_bitwidth))

    # List of keys to extract data for #NOTE TODO add support for more architectures!
    if "simba" in architecture:
        keys_to_extract = ["PEWeightRegs", "PEAccuBuffer", "PEWeightBuffer", "PEInputBuffer", "GlobalBuffer", "DRAM"]
    else:  # assume eyeriss
        keys_to_extract = ["psum_spad", "weights_spad", "ifmap_spad", "shared_glb", "DRAM"]

    # Extract data for each key and add it to the result_dict
    for key in keys_to_extract:
        extracted_data = extract_scalar_accesses_data(data, key)
        if extracted_data:
            result_dict[f"{key} data"] = extracted_data
    return result_dict


def parse_timeloop_stats(filename: str) -> Dict[str, Any]:
    """
    Parses statistics from a Timeloop XML file.
    Modified code originating from: https://github.com/NVlabs/timeloop/blob/master/scripts/parse_timeloop_output.py

    Args:
        filename (str): The path to the Timeloop XML file.

    Returns:
        Dict[str, Any]: A dictionary containing parsed statistics and metrics.
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    # Parse out the problem shape
    problem_dims = root.findall('a')[0].findall('workload_')[0].findall('factorized_bounds_')[0].findall('item')
    problem = [int(pd.findall('second')[0].text) for pd in problem_dims]  # FIXedME generalize for non-conv problems

    macs = np.prod(problem)

    topology = root.findall('engine')[0].findall('topology_')[0]

    # Get the list of storage/arithmetic levels
    levels = topology.findall('levels_')[0]
    num_levels = int(levels.findall('count')[0].text)
    level_ptrs = levels.findall('item')

    # Get the list of networks
    networks = topology.findall('networks_')[0]
    num_networks = int(networks.findall('count')[0].text)
    network_ptrs = networks.findall('item')

    # Initialize a dictionary that stores energy breakdown and other statistics
    energy_breakdown_pJ = {}
    arithmetic_level_found = False

    for level_id in range(len(level_ptrs)):
        level_ptr = level_ptrs[level_id]
        level = level_ptr.findall('px')[0]

        # The XML structure is interesting. Every Level gets a <px>, but
        # only the first object of each type gets a full class_id descriptor.
        # For example, the first model::BufferLevel item will get:
        #    <px class_id="9" class_name="model::BufferLevel" tracking_level="1" version="0" object_id="_1">
        # but subsequent levels will get something like:
        # <px class_id_reference="9" object_id="_2">
        # with increasing object_ids. We can keep a table of new class_ids as
        # we encounter them, but for now we'll just hack something that works.

        # Is this the Arithmetic level (the only one)?
        if 'class_id' in level.attrib and level.attrib['class_name'] == "model::ArithmeticUnits":
            assert arithmetic_level_found is False
            arithmetic_level_found = True
            cycles = int(level.findall('cycles_')[0].text)
            utilized_instances = float(level.findall('utilized_instances_')[0].text)
            total_instances_list = level.findall('specs_')[0].findall('instances')[0].findall('t_')
            if total_instances_list == []:  # this happens when no mapping is returned by timeloop
                total_instances = 1  # dummy value
            else:
                total_instances = float(level.findall('specs_')[0].findall('instances')[0].findall('t_')[0].text)
            arithmetic_utilization = utilized_instances/total_instances
            energy_breakdown_pJ['MAC'] = {'energy': float(level.findall('energy_')[0].text), 'utilization': arithmetic_utilization}
            continue

        # If we are here, we are not an arithmetic level.

        # Level specifications and stats.
        specs = level.findall('specs_')[0]
        stats = level.findall('stats_')[0]

        generic_level_specs = specs.findall('LevelSpecs')[0]
        level_name = generic_level_specs.findall('level_name')[0].text

        # Storage access energy
        reads_per_instance = get_stat(stats, 'reads', int)
        updates_per_instance = get_stat(stats, 'updates', int)
        fills_per_instance = get_stat(stats, 'fills', int)
        accesses_per_instance = reads_per_instance + updates_per_instance + fills_per_instance

        utilized_capacity = get_stat(stats, 'utilized_capacity', int)
        try:
            instances = get_stat(stats, 'utilized_instances', int)
        except ValueError:
            instances = get_stat(stats, 'utilized_instances', float)
        clusters = get_stat(stats, 'utilized_clusters', int)

        total_instances_obj = specs.findall('instances')[0].findall('t_')
        if len(total_instances_obj) == 0:
            total_instances = sum(instances)
        else:
            total_instances = int(total_instances_obj[0].text)

        total_capacity_obj = specs.findall('size')[0].findall('t_')
        if len(total_capacity_obj) == 0:
            total_capacity = sum(utilized_capacity)
        else:
            total_capacity = int(total_capacity_obj[0].text)

        energy_per_access_per_instance = get_stat(stats, 'energy_per_access', float)
        storage_access_energy_in_pJ = energy_per_access_per_instance * accesses_per_instance * instances
        read_energy = energy_per_access_per_instance * reads_per_instance * instances

        # Find read-network connected to this storage level by looking at the first word
        # in the network's name.
        # FIXME: all this ugliness is because of legacy topology structure. We should
        # simply report networks independently.
        assert(level_id >= 1)
        for n in network_ptrs:
            network_name = n.findall('first')[0].text
            network_source = network_name.split(None, 1)[0]
            if network_source == level_name:
                network = n.findall('second')[0].findall('px')[0]
                break
        # network_ptr = network_ptrs[level_id-1]
        # network = network_ptr.findall('second')[0].findall('px')[0]

        # Network energy
        # network = level.findall('network_')[0]
        network_stats = network.findall('stats_')[0]

        # FIXedME when router energy !== zero, need to fetch total energy per instance
        num_hops = get_stat(network_stats, 'num_hops', float)
        energy_per_hop_per_instance = get_stat(network_stats, 'energy_per_hop', float)
        ingresses = 0  # get_stat(network_stats, 'ingresses', int)
        network_energy_per_instance_pJ = get_stat(network_stats, 'energy', float)
        network_energy_in_pJ = network_energy_per_instance_pJ * instances

        # Add multicast factors
        multicast = get_stat(network_stats, 'multicast_factor', int)
        dist_multicast = get_stat(network_stats, 'distributed_multicast', int)

        # Add energy
        spatial_add_energy_per_instance = get_stat(network_stats, 'spatial_reduction_energy', float)
        temporal_add_energy_per_instance = get_stat(stats, 'temporal_reduction_energy', float)
        temporal_add_energy = np.nansum(temporal_add_energy_per_instance * instances)
        spatial_add_energy = np.nansum(spatial_add_energy_per_instance * instances)

        # Address generation energy
        address_generation_energy_per_cluster = get_stat(stats, 'addr_gen_energy', float)
        address_generation_energy = np.nansum(address_generation_energy_per_cluster * clusters)

        # Special Case when the memory level is a dummy (capacity = 0)
        if total_capacity == 0:
            utilization = 0
        else:
            utilization = sum((utilized_capacity*instances)/(total_capacity*total_instances))

        energy_breakdown_pJ[level_name] = {
            'energy': np.nansum(storage_access_energy_in_pJ) + np.nansum(network_energy_in_pJ) + temporal_add_energy + spatial_add_energy + address_generation_energy,
            'storage_access_energy': np.nansum(storage_access_energy_in_pJ),
            'read_energy': np.nansum(read_energy),
            'temporal_add_energy': temporal_add_energy,
            'spatial_add_energy': spatial_add_energy,
            'address_generation_energy': address_generation_energy,
            'network_energy': np.nansum(network_energy_in_pJ),
            'energy_per_access_per_instance': energy_per_access_per_instance,
            'reads_per_instance': reads_per_instance,
            'updates_per_instance': updates_per_instance,
            'fills_per_instance': fills_per_instance,
            'accesses_per_instance': accesses_per_instance,
            'instances': instances,
            'utilization': utilization,
            'multicast': multicast,
            'dist_multicast': dist_multicast,
            'num_hops': num_hops,
            'ingresses': ingresses,
            'energy_per_hop_per_instance': energy_per_hop_per_instance
        }

    energy_pJ = sum([value['energy'] for key, value in energy_breakdown_pJ.items()])

    # Crude check to find out if timeloop produced an output.
    if arithmetic_level_found:
        output = {
            'problem': problem,
            'utilization': arithmetic_utilization,
            'cycles': cycles,
            'energy_pJ': energy_pJ,
            'energy_per_mac': energy_pJ/macs,
            'macs': macs,
            'energy_breakdown_pJ': energy_breakdown_pJ
        }
    else:
        output = {}

    return output


def parse_experiments_json(filename: str, result_dict: dict, experiments: list = [], verbose: bool = False) -> Dict[str, Any]:
    """
    Parses experiments from a JSON file and updates the result dictionary with parsed data.

    Args:
        filename (str): Path to the JSON file.
        result_dict (dict): The dictionary to update with parsed data.
        experiments (list, optional): A list of experiments to parse. Defaults to an empty list.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
        Dict[str, Any]: Updated result dictionary with parsed data.
    """
    result_dict["xml_data"] = []

    parsed_output = parse_timeloop_stats(filename)
    result_dict["xml_data"].append(parsed_output)

    return result_dict


class MapperFacade:
    """Class represents the facade interface for calling timeloop mapper and retrieve hardware metrics.

    The __init__ method of this class takes in the relative path to the timeloop configs folder
    and the name of the HW architecture to be used for invoking the subsequent timeloop-mapper calls.

    Args:
        configs_rel_path (str): Relative path to the timeloop configs folder.
        architecture (str): Name of the architecture to be used along with its associated components and constraints.
    """
    def __init__(self, configs_rel_path: str = "timeloop_utils/timeloop_configs", architecture: str = "eyeriss") -> None:
        self._architecture = architecture
        self._mode = f"timeloop-mapper"
        self._thread_id = threading.get_ident()

        self.configs_path = configs_rel_path
        self.arch = glob.glob(f"{self.configs_path}/architectures/{architecture}/*.yaml")[0]
        self.components = glob.glob(f"{self.configs_path}/architectures/{architecture}/components/*.yaml")
        self.constraints = glob.glob(f"{self.configs_path}/architectures/{architecture}/constraints/*.yaml")
        self.sparse_opt = glob.glob(f"{self.configs_path}/architectures/{architecture}/sparse_opt/*.yaml")

    def _load_cache(self, cache_file_path: str) -> Dict[str, Any]:
        """
        Load the cache from a gzip-compressed JSON file.

        Args:
            cache_file_path (str): The file path to the gzip-compressed JSON cache file.

        Returns:
            Dict[str, Any]: The loaded cache data as a dictionary. If the file does not exist, returns an empty dictionary.
        """
        if os.path.exists(cache_file_path):
            with gzip.open(cache_file_path, 'rt') as file:
                return json.load(file)
        return {}

    def _save_cache(self, cache: Dict[str, Any], cache_file_path: str) -> None:
        """
        Save the cache to a gzip-compressed JSON file.

        Args:
            cache (Dict[str, Any]): The cache data to be saved.
            cache_file_path (str): The file path where the gzip-compressed JSON cache file will be saved.
        """
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

        with gzip.open(cache_file_path, 'wt') as file:
            json.dump(cache, file, indent=2, cls=JSONEncoder)

    def _modify_mapper_configs(self, mapper_config: Dict[str, Any], heuristic: str, metrics: Tuple[str, str], threads: Union[str, int], total_valid: int, log_all: bool) -> Dict[str, Any]:
        """
        Modifies the mapper configuration based on specified settings.

        Args:
            mapper_config (Dict[str, Any]): The original (mapper) configuration dictionary.
            heuristic (str): The heuristic type to use ('exhaustive', 'hybrid', 'linear', 'random').
            metrics (Tuple[str, str]): A tuple of metrics to optimize for.
            threads (Union[str, int]): The number of threads to use, or 'all' for all available threads. A value of 0 means that this criteria is not used for thread termination. Defaults to 0.
            total_valid (int): The number of total valid mappings to consider across all available mapper threads.
            log_all (bool): Flag to enable logging of all mappings.

        Returns:
            Dict[str, Any]: The modified mapper configuration dictionary.
        """
        mapper_config["mapper"]["out_prefix"] = f"{self._mode}_{self._thread_id}"
        mapper_config["mapper"]["optimization-metrics"] = list(metrics) if metrics[1] else [metrics[0]]
        mapper_config["mapper"]["search-size"] = total_valid

        if log_all:
            mapper_config["mapper"]["log-oaves"] = True
            mapper_config["mapper"]["log-suboptimal"] = True
            mapper_config["mapper"]["log-all"] = True
            mapper_config["mapper"]["log-stats"] = True

        if isinstance(threads, int):
            mapper_config["mapper"]["num-threads"] = threads

        if heuristic == "exhaustive":
            mapper_config["mapper"]["algorithm"] = "linear-pruned"
            mapper_config["mapper"]["victory-condition"] = 0
            mapper_config["mapper"]["timeout"] = 0
            # Remove the "max-permutations-per-if-visit" if it exists
            mapper_config["mapper"].pop("max-permutations-per-if-visit", None)
        else:
            if total_valid == 0:
                mapper_config["mapper"]["victory-condition"] = 500
                mapper_config["mapper"]["max-permutations-per-if-visit"] = 16
                mapper_config["mapper"]["timeout"] = 15000
            else:
                mapper_config["mapper"].pop("victory-condition", None)
                mapper_config["mapper"].pop("max-permutations-per-if-visit", None)
                mapper_config["mapper"].pop("timeout", None)

            if heuristic == "random":
                mapper_config["mapper"]["algorithm"] = "random-pruned"
            elif heuristic == "linear":
                mapper_config["mapper"]["algorithm"] = "linear-pruned"
            elif heuristic == "hybrid":
                mapper_config["mapper"]["algorithm"] = "hybrid"

        return mapper_config

    def run_one_workload(self, workload: str, bitwidth: str, batch_size: int = 1, threads: Union[str, int] = "all", heuristic: str = "random", metrics: Tuple[str, str] = ("edp", ""), total_valid: int = 0, out_dir: str = "tmp_outputs", cache_dir: str = "timeloop_mapper_cache", cache_name: str = "cache", log_all: bool = False, verbose: bool = False, clean: bool = True) -> Dict[str, Any]:
        """
        Runs the mapper on a single workload.

        Args:
            workload (str): Relative path to the workload configuration file.
            bitwidth (str): The bitwidth configuration.
            batch_size (int): The batch size to use. Defaults to 1.
            threads (Union[str, int]): The number of threads to use, or 'all' for all available threads. Defaults to "all".
            heuristic (str): The heuristic type to use ('exhaustive', 'hybrid', 'linear', 'random'). Defaults to "random".
            metrics (Tuple[str, str]): A tuple of metrics to optimize for. Possible values are all six combinations of `energy`, `delay`, `lla` with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("edp", "").
            total_valid (int): The number of total valid mappings to consider across all available mapper threads. A value of 0 means that this criteria is not used for thread termination. Defaults to 0.
            out_dir (str): Relative path to the directory to store timeloop-mapper output files. Defaults to "tmp_outputs".
            cache_dir (str): Relative path to the cache directory where the timeloop-mapper cache file is stored. Defaults to "timeloop_mapper_cache".
            cache_name (str): Name of the JSON cache file to store the results. Defaults to "cache".
            log_all (bool): Flag to enable logging of all mappings. Defaults to False.
            verbose (bool): Flag to enable printing the timeloop-mapper output. Defaults to False.
            clean (bool): Flag to clean up temporary files after execution. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the best mapping's hardware parameters and total runtime of timeloop-mapper call.
        """
        mapper = f"{self.configs_path}/mapper_heuristics/mapper_template.yaml"

        cache_file_path = os.path.join(cache_dir, f"{cache_name}.json.gz")
        cache = self._load_cache(cache_file_path)
        layer = workload.split("/")[-1].split(".")[0]

        if layer in cache:
            if bitwidth in cache[layer]:
                # Return dictionary with the best found HW params and total mapper runtime from cache
                return cache[layer][bitwidth]
        else:
            cache[layer] = {}  # Initialize cache[layer] as a dictionary

        with open(mapper, "r") as map:
            try:
                config_dict = yaml.safe_load(map)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)

        # Modify the mapper heuristic settings for the given settings
        config_dict = self._modify_mapper_configs(config_dict, heuristic, metrics, threads, total_valid, log_all)

        # Write the modified YAML data to a temporary file
        modified_mapper = "_".join(os.path.splitext(mapper)[0].split('_')[:-1]) + f"_{self._thread_id}.yaml"
        with open(modified_mapper, "w") as modified_map:
            yaml.dump(config_dict, modified_map)

        start_time = time.time()
        tmp_dir = f"{out_dir}_{self._thread_id}"
        os.makedirs(tmp_dir, exist_ok=True)

        # Running the timeloop-mapper for the given workload and chosen mapper heuristic settings
        if verbose:
            subprocess.run([self._mode, self.arch] + self.components + self.constraints + self.sparse_opt
                           + [modified_mapper, workload, "-o", tmp_dir], check=True)
        else:
            subprocess.run([self._mode, self.arch] + self.components + self.constraints + self.sparse_opt
                           + [modified_mapper, workload, "-o", tmp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Reading the CSV file into a dictionary
        with open(f"{tmp_dir}/{self._mode}_{self._thread_id}.stats.csv", "r") as f:
            reader = csv.DictReader(f)
            result_dict = next(reader)

        # Read the content of the text file to retrieve the total scalar accesses and Op per Byte
        with open(f"{tmp_dir}/{self._mode}_{self._thread_id}.stats.txt", "r") as f:
            data = f.read()
            # Add the total scalar accesses, Op per Byte and memory size stats to the result dictionary
            result_dict = extract_memory_stats(self._architecture, workload, data, result_dict)

        # Read the content of the text file to retrieve the total scalar accesses and Op per Byte
        file_paths = glob.glob(os.path.join(tmp_dir, '*map+stats.xml'))

        # Add the xml data to the result dictionary
        result_dict = parse_experiments_json(file_paths[0], result_dict=result_dict)

        # Calculate weights_memory_size
        with open(workload, 'r') as work_file:
            work_data = yaml.safe_load(work_file)

        # Deleting the tmp files
        if clean:
            shutil.rmtree(tmp_dir)
            os.remove(modified_mapper)
            [os.remove(f) for f in glob.glob('./*.log')]

        end_time = time.time()
        runtime = end_time - start_time
        threads = threads if threads != "all" else multiprocessing.cpu_count()

        cache[layer][bitwidth] = {"Mode": self._mode, "HW": self._architecture, "Workload": layer, "Bitwidths": bitwidth, "Batch_size": batch_size, "Mapper heuristic": heuristic, "Total valid": total_valid, "Threads": threads, "Optimized_metric_1": metrics[0], "Optimized_metric_2": metrics[1], **result_dict, "Runtime [s]": "{:.2f}".format(runtime)}
        self._save_cache(cache, cache_file_path)

        # Return dictionary with the best found HW params and total mapper runtime
        return cache[layer][bitwidth]

    def run_all_workloads(self, workloads: str, batch_size: int = 1, bitwidths: Optional[Union[Tuple[int, int, int], Dict[str, Dict[str, int]]]] = None, threads: Union[str, int] = "all", heuristic: str = "random", metrics: Tuple[str, str] = ("edp", ""), total_valid: int = 0, out_dir: str = "tmp_outputs", cache_dir: str = "timeloop_mapper_cache", cache_name: str = "cache", log_all: bool = False, verbose: bool = False, clean: bool = True) -> Dict[str, Any]:
        """
        Runs timeloop-mapper for all workloads (i.e. a CNN network's layers) in a given folder with specified mapper settings.

        Args:
            workloads (str): Relative path to the folder containing the workload configuration files.
            batch_size (int): The batch size for the model. Defaults to 1.
            bitwidths (Optional[Union[Tuple[int, int, int], Dict[str, Dict[str, int]]]]): The bitwidth settings for the model's workloads. Can be None for native settings, a tuple (i.e. (8,4,8) for uniform settings across layers, or a dictionary for non-uniform settings per layer (for example: `{"layer_1": {"Inputs": 8, "Weights": 4, "Outputs": 6},"layer_2": {"Inputs": 6, "Weights": 2, "Outputs": 5}}`). Defaults to None.
            threads (Union[str, int]): The number of threads to use for the mapper heuristics, or 'all' for all available threads. Defaults to "all".
            heuristic (str): The heuristic type to use for the mapper. Choices are `exhaustive`, `hybrid`, `linear` or `random`. Defaults to "random".
            metrics (Tuple[str, str]): A tuple of two metrics to optimize for. Possible values are all six combinations of `energy`, `delay`, `lla` with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("edp", "").
            total_valid (int): The number of total valid mappings to consider across all available mapper threads. A value of 0 means that this criteria is not used for thread termination. Defaults to 0.
            out_dir (str): Relative path to the output directory for the timeloop-mapper's output files. Defaults to "tmp_outputs".
            cache_dir (str): Relative path to the cache directory where the timeloop-mapper cache file is stored. Defaults to "timeloop_mapper_cache".
            cache_name (str): Name of the JSON cache file to store the results. Defaults to "cache".
            log_all (bool): Whether to log all mappings. Defaults to False.
            verbose (bool): Whether to print the timeloop-mapper output. Defaults to False.
            clean (bool): Flag to delete the temporary files generated by timeloop-mapper. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary containing the best mappings HW parameters and total runtime of the individual workloads timeloop-mapper calls.
        """
        workloads = glob.glob(f"{workloads}/*.yaml")
        hw_params = {}

        # Retrieve parameters for each workload
        for i, workload in enumerate(workloads):
            if bitwidths is None:
                bitwidth = "native_native_native"
            elif isinstance(bitwidths, tuple):
                bitwidth = f"{bitwidths[0]}_{bitwidths[1]}_{bitwidths[2]}"
            else:
                key = list(bitwidths.keys())[i]
                bitwidth = f"{bitwidths[key]['Inputs']}_{bitwidths[key]['Weights']}_{bitwidths[key]['Outputs']}"
            hw_params[workload.split("/")[-1].split(".")[0]] = self.run_one_workload(workload=workload, batch_size=batch_size, bitwidth=bitwidth, threads=threads, heuristic=heuristic, metrics=metrics, total_valid=total_valid, out_dir=f"{out_dir}/{workload.split('/')[-1].split('.')[0]}", cache_dir=cache_dir, cache_name=cache_name, log_all=log_all, verbose=verbose, clean=clean)
            print("Finished workload ", i+1, "/", len(workloads))

        # Return dictionary with individual workload's HW params and runtime
        return hw_params

    def get_hw_params_create_model(self, framework: str, model: str, num_classes: int = 1000, batch_size: int = 1, bitwidths: Optional[Union[Tuple[int, int, int], Dict[str, Dict[str, int]]]] = None, input_size: str = "224,224,3", threads: Union[str, int] = "all", heuristic: str = "random", metrics: Tuple[str, str] = ("edp", ""), total_valid: int = 0, out_dir: str = "tmp_outputs", cache_dir: str = "timeloop_mapper_cache", cache_name: str = "cache", log_all: bool = False, verbose: bool = False, clean: bool = True) -> Dict[str, Any]:
        """
        Creates a CNN model and runs timeloop-mapper on all its workloads (i.e. a CNN network's layers) with specified mapper settings.

        Args:
            framework (str): Choice of api ('pytorch' or 'keras').
            model (str): Model to be instantiated.
            num_classes (int): Number of classes for the classification task. Defaults to 1000.
            batch_size (int): The batch size for the model. Defaults to 1.
            bitwidths (Optional[Union[Tuple[int, int, int], Dict[str, Dict[str, int]]]]): The bitwidth settings for the model's workloads. Can be None for native settings, a tuple (i.e. (8,4,8) for uniform settings across layers, or a dictionary for non-uniform settings per layer (for example: `{"layer_1": {"Inputs": 8, "Weights": 4, "Outputs": 6},"layer_2": {"Inputs": 6, "Weights": 2, "Outputs": 5}}`). Defaults to None.
            input_size (str): Input size of the model. Defaults to "224,224,3".
            threads (Union[str, int]): The number of threads to use for the mapper heuristics, or 'all' for all available threads. Defaults to "all".
            heuristic (str): The heuristic type to use for the mapper. Choices are `exhaustive`, `hybrid`, `linear` or `random`. Defaults to "random".
            metrics (Tuple[str, str]): A tuple of two metrics to optimize for. Possible values are all six combinations of `energy`, `delay`, `lla` with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("edp", "").
            total_valid (int): The number of total valid mappings to consider across all available mapper threads. A value of 0 means that this criteria is not used for thread termination. Defaults to 0.
            out_dir (str): Relative path to the output directory for the timeloop-mapper's output files. Defaults to "tmp_outputs".
            cache_dir (str): Relative path to the cache directory where the timeloop-mapper cache file is stored. Defaults to "timeloop_mapper_cache".
            cache_name (str): Name of the JSON cache file to store the results. Defaults to "cache".
            log_all (bool): Whether to log all mappings. Defaults to False.
            verbose (bool): Whether to print the timeloop-mapper output. Defaults to False.
            clean (bool): Flag to delete the temporary files generated by timeloop-mapper. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary with hardware parameters and runtime for each workload of the created model.
        """
        if isinstance(input_size, str):
            input_size = tuple((int(d) for d in str.split(input_size, ",")))
        # Create templates for individual model's CONV layers
        if framework == 'pytorch':
            create_pytorch_model(model_name=model, input_size=input_size, batch_size=batch_size, out_dir="timeloop_utils/construct_workloads/parsed_models", out_file=model, num_classes=num_classes, verbose=verbose)
        elif framework == 'keras':
            create_keras_model(model_name=model, input_size=input_size, batch_size=batch_size, out_dir="timeloop_utils/construct_workloads/parsed_models", out_file=model, num_classes=num_classes, verbose=verbose)
        else:
            raise ValueError("Unsupported framework. Choose 'keras' or 'pytorch'.")

        # Construct timeloop workloads from the created templates and add to them the bitwidth settings
        yaml_model = f"timeloop_utils/construct_workloads/parsed_models/{model.split('/')[-1].split('.')[0]}.yaml"
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        workloads_location = f"{self.configs_path}/workload_shapes/{timestamp}"

        if bitwidths is None:
            construct_workloads(model=yaml_model, bitwidth_setting="native", uniform_width_set=None, non_uniform_width_set=None, out_dir=workloads_location, out_file=model, verbose=verbose)
        elif isinstance(bitwidths, dict):
            construct_workloads(model=yaml_model, bitwidth_setting="non-uniform", uniform_width_set=None, non_uniform_width_set=bitwidths, out_dir=workloads_location, out_file=model, verbose=verbose)
        elif isinstance(bitwidths, tuple) and len(bitwidths) == 3 and all(isinstance(item, int) for item in bitwidths):
            construct_workloads(model=yaml_model, bitwidth_setting="uniform", uniform_width_set=bitwidths, non_uniform_width_set=None, out_dir=workloads_location, out_file=model, verbose=verbose)
        else:
            print("Unrecognized bitwidths object. Expected dict, tuple or None to represent non-uniform, uniform and native bitwidhts settings, respectively.", file=sys.stderr)
            sys.exit(0)

        # Run timeloop-mapper on the created workloads
        results = self.run_all_workloads(workloads=workloads_location, batch_size=batch_size, bitwidths=bitwidths, threads=threads, heuristic=heuristic, metrics=metrics, total_valid=total_valid, out_dir=out_dir, cache_dir=cache_dir, cache_name=cache_name, log_all=log_all, verbose=verbose, clean=clean)
        # Clean up created workload_shapes files
        if clean:
            shutil.rmtree(workloads_location)
        return results

    def get_hw_params_parse_model(self, framework: str, model: str, arch: str, batch_size: int = 1, bitwidths: Optional[Union[Tuple[int, int, int], Dict[str, Dict[str, int]]]] = None, input_size: str = "224,224,3", threads: Union[str, int] = "all", heuristic: str = "random", metrics: Tuple[str, str] = ("edp", ""), total_valid: int = 0, out_dir: str = "tmp_outputs", cache_dir: str = "timeloop_mapper_cache", cache_name: str = "cache", log_all: bool = False, verbose: bool = False, clean: bool = True) -> Dict[str, Any]:
        """
        Parses a CNN model and runs timeloop-mapper on all its workloads (i.e. a CNN network's layers) with specified mapper settings.

        Args:
            framework (str): Choice of api ('pytorch' or 'keras').
            model (str): Path to the CNN model or state_dict to be parsed.
            arch (str): Name of the PyTorch model (custom or torchvision) to be instantiated for the parsed model if only state_dict is provided.
            batch_size (int): The batch size for the model. Defaults to 1.
            bitwidths (Optional[Union[Tuple[int, int, int], Dict[str, Dict[str, int]]]]): The bitwidth settings for the model's workloads. Can be None for native settings, a tuple (i.e. (8,4,8) for uniform settings across layers, or a dictionary for non-uniform settings per layer (for example: `{"layer_1": {"Inputs": 8, "Weights": 4, "Outputs": 6},"layer_2": {"Inputs": 6, "Weights": 2, "Outputs": 5}}`). Defaults to None.
            input_size (str): Input size of the model. Defaults to "224,224,3".
            threads (Union[str, int]): The number of threads to use for the mapper heuristics, or 'all' for all available threads. Defaults to "all".
            heuristic (str): The heuristic type to use for the mapper. Choices are `exhaustive`, `hybrid`, `linear` or `random`. Defaults to "random".
            metrics (Tuple[str, str]): A tuple of two metrics to optimize for. Possible values are all six combinations of `energy`, `delay`, `lla` with an additional seventh option `edp`, leaving the second metric blank. Defaults to ("edp", "").
            total_valid (int): The number of total valid mappings to consider across all available mapper threads. A value of 0 means that this criteria is not used for thread termination. Defaults to 0.
            out_dir (str): Relative path to the output directory for the timeloop-mapper's output files. Defaults to "tmp_outputs".
            cache_dir (str): Relative path to the cache directory where the timeloop-mapper cache file is stored. Defaults to "timeloop_mapper_cache".
            cache_name (str): Name of the JSON cache file to store the results. Defaults to "cache".
            log_all (bool): Whether to log all mappings. Defaults to False.
            verbose (bool): Whether to print the timeloop-mapper output. Defaults to False.
            clean (bool): Flag to delete the temporary files generated by timeloop-mapper. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary with hardware parameters and runtime for each workload of the parsed model.
        """
        if not os.path.exists(model):
            raise FileNotFoundError(f"No model file `{model}` found.")
        
        if framework == 'pytorch':
            # Ensure the model file is a PyTorch model
            model_parts = model.split("/")[-1].split(".")
            if len(model_parts) > 2:
                model_ext = ".".join(model_parts[-2:])
            else:
                model_ext = model_parts[-1]
            assert model_ext in ["pth", "pt", "pth.tar", "pt.tar"], "Unrecognized model file extension. Expected .pt, .pth, .pt.tar or .pth.tar for PyTorch model."
            parse_pytorch_model(model_file=model, input_size=input_size, batch_size=batch_size, out_dir="timeloop_utils/construct_workloads/parsed_models", out_file=model.split("/")[-1].split(".")[0], architecture=arch, verbose=verbose)
        elif framework == 'keras':
            # Ensure the model file is a Keras model
            model_parts = model.split("/")[-1].split(".")
            if len(model_parts) > 2:
                model_ext = ".".join(model_parts[-2:])
            else:
                model_ext = model_parts[-1]
            assert model_ext in ["h5", "hdf5", "keras", "tf"], "Unrecognized model file extension. Expected .h5, .hdf5, .keras, or .tf for Keras model."
            parse_keras_model(model_file=model, input_size=input_size, batch_size=batch_size, out_dir="timeloop_utils/construct_workloads/parsed_models", out_file=model.split("/")[-1].split(".")[0], verbose=verbose)
        else:
            raise ValueError("Unsupported framework. Choose 'keras' or 'pytorch'.")


        # Create templates for individual model's CONV layers
        parse_pytorch_model(model_file=model, input_size=input_size, batch_size=batch_size, out_dir="timeloop_utils/construct_workloads/parsed_models", out_file=model.split("/")[-1].split(".")[0], architecture=arch, verbose=verbose)

        # Construct timeloop workloads from the created templates and add to them the bitwidth settings
        yaml_model = f"timeloop_utils/construct_workloads/parsed_models/{model.split('/')[-1].split('.')[0]}.yaml"
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        workloads_location = f"{self.configs_path}/workload_shapes/{timestamp}"

        if bitwidths is None:
            construct_workloads(model=yaml_model, bitwidth_setting="native", uniform_width_set=None, non_uniform_width_set=None, out_dir=workloads_location, out_file=arch, verbose=verbose)
        elif isinstance(bitwidths, dict):
            construct_workloads(model=yaml_model, bitwidth_setting="non-uniform", uniform_width_set=None, non_uniform_width_set=bitwidths, out_dir=workloads_location, out_file=arch, verbose=verbose)
        elif isinstance(bitwidths, tuple) and len(bitwidths) == 3 and all(isinstance(item, int) for item in bitwidths):
            construct_workloads(model=yaml_model, bitwidth_setting="uniform", uniform_width_set=bitwidths, non_uniform_width_set=None, out_dir=workloads_location, out_file=arch, verbose=verbose)
        else:
            print("Unrecognized bitwidths object. Expected dict, tuple or None to represent non-uniform, uniform and native bitwidhts settings, respectively.", file=sys.stderr)
            sys.exit(0)

        # Run timeloop-mapper on the created workloads
        results = self.run_all_workloads(workloads=workloads_location, batch_size=batch_size, bitwidths=bitwidths, threads=threads, heuristic=heuristic, metrics=metrics, total_valid=total_valid, out_dir=out_dir, cache_dir=cache_dir, cache_name=cache_name, log_all=log_all, verbose=verbose, clean=clean)
        # Clean up created workload_shapes files
        if clean:
            shutil.rmtree(workloads_location)
        return results


if __name__ == "__main__":
    # For example runs
    facade = MapperFacade()

    # Example usage run creating and evaluating workloads for alexnet pytorch model with no quantization (bitwidths=None) for 100 total valid mappings per layer (for fast search)
    results = facade.get_hw_params_create_model(framework="pytorch", model="vgg16", batch_size=1, bitwidths=None, input_size="224,224,3", threads="all", heuristic="random", metrics=("edp", ""), cache_dir="vgg16_torch", total_valid=100)
    dict_to_json(results, "res_native_vgg16_torch.json")

    # Example usage run creating and evaluating workloads for alexnet keras model with no quantization (bitwidths=None) for 100 total valid mappings per layer (for fast search)
    results = facade.get_hw_params_create_model(framework="keras", model="VGG16", batch_size=1, bitwidths=None, input_size="224,224,3", threads="all", heuristic="random", metrics=("edp", ""), cache_dir="vgg16_keras", total_valid=100)
    dict_to_json(results, "res_native_vgg16_keras.json")

    # Example usage run creating and evaluating workloads for alexnet pytorch model (classifying 1000 classes) with no quantization (bitwidths=None)
    """
    results = facade.get_hw_params_create_model(framework="pytorch", model="alexnet", num_classes=1000, batch_size=1, bitwidths=None, input_size="224,224,3", threads="all", heuristic="exhaustive", metrics=("edp", ""), cache_dir="an_run_1", clean=True)
    dict_to_json(results, "results_native_alexnet.json")
    """

    # Example usage run creating and evaluating workloads for alexnet pytorch model (classifying 1000 classes) with uniform quantization for each layer (bitwidths=(8,4,8))
    """
    results = facade.get_hw_params_create_model(framework="pytorch", model="alexnet", num_classes=1000, batch_size=1, bitwidths=(8,4,8), input_size="224,224,3", threads="all", heuristic="exhaustive", metrics=("edp", ""), cache_dir="an_run_2", clean=True)
    dict_to_json(results, "results_uniform_alexnet.json")
    """

    # Example usage run creating and evaluating workloads for custom user made alexnet pytorch model with non-uniform quantization for each layer (bitwidths=dict)
    """
    json_dict = json_file_to_dict("timeloop_utils/construct_workloads/temps/bitwidths_alexnet_sample.json")
    results = facade.get_hw_params_create_model(framework="pytorch", model="alexnet", num_classes=1000, batch_size=1, bitwidths=json_dict, input_size="224,224,3", threads="all", heuristic="random", metrics=("edp", ""), cache_dir="an_run_3", clean=True)
    dict_to_json(results, "results_non_uniform_alexnet.json")
    """
    
    # If you have Keras/Pytorch model, you can use the get_hw_params_parse_model() function instead.
    
