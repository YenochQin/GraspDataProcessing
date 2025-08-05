#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id : CSFs_descriptor_parallel.py
@date : 2025/08/05
@author : YenochQin (秦毅)
@description : Use multiprocessing to generate CSF descriptors in parallel.
                This script is self-contained and does not require external data modules.
'''
import multiprocessing
import re
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm

# --- Placeholder for data structures ---
# In a real scenario, this class would be imported. For this self-contained script,
# we define a basic structure to hold the necessary data.
class CSFs:
    """
    A placeholder class to represent the structure of CSF data.
    In the actual project, this would be a more complex data module.
    """
    def __init__(self, subshell_info_raw: List[str], csfs_block_data: List[List[List[str]]]):
        self.subshell_info_raw = subshell_info_raw
        self.CSFs_block_data = csfs_block_data

# --- Utility and Helper Functions ---
# These functions are dependencies copied from other modules to make this script standalone.

def chunk_string(s: str, n: int) -> list[str]:
    """Splits a string into fixed-length chunks."""
    return [s[i:i+n] for i in range(0, len(s), n)]

def str_subshell_2_kappa(str_subshell: str) -> int:
    r'''
    Converts subshell string to kappa value.
    j = l + 1/2, kappa = -(l+1)
    j = l - 1/2, kappa = +l 
    '''
    kappa_value = {
        "s ": -1, "p-":  1, "p ": -2, "d-":  2, "d ": -3,
        "f-":  3, "f ": -4, "g-":  4, "g ": -5, "h-":  5,
        "h ": -6, "i-":  6, "i ": -7, "j-":  7, "j ": -8,
        "k-":  8, "k ": -9
    }
    return kappa_value.get(str_subshell, 0)

def J_to_doubleJ(J_str: str) -> int:
    """
    Converts a J-value string (e.g., '3/2', '2') to its double value (2J) as an integer.
    """
    J_str = J_str.strip()
    if '/' in J_str:
        numerator, _ = map(int, J_str.split('/'))
        return numerator
    else:
        return int(J_str) * 2

def if_subshell_full_charged(subshell_name: str, subshell_charged_num: int) -> bool:
    """Checks if a subshell is fully charged with electrons."""
    full_charged = {
        "s ": 2, "p-": 2, "p ": 4, "d-": 4, "d ": 6,
        "f-": 6, "f ": 8, "g-": 8, "g ": 10,
    }
    # Normalize subshell name by ensuring it has a trailing space if it's a single letter
    if len(subshell_name) == 1 and subshell_name.isalpha():
        subshell_name += " "
    return full_charged.get(subshell_name, 0) == subshell_charged_num

def get_CSFs_peel_subshells(CSFs_file_data: CSFs) -> List[str]:
    """Extracts the list of peel subshells from the CSF data object."""
    peel_subshells_line = CSFs_file_data.subshell_info_raw[-1].strip()
    return [s.strip() for s in peel_subshells_line.split() if s.strip()]

# --- Core CSF Parsing Functions ---

def parse_csf_2_descriptor(peel_subshells_List: List[str], csf: List[str]) -> np.ndarray:
    """
    Parses a single CSF into a descriptor array (3 features per orbital).
    Features: [electron_count, intermediate_J, coupled_J]
    """
    subshells_line, middle_line_raw, coupling_line_raw = [line.rstrip() for line in csf]
    line_length = len(subshells_line)
    middle_line = middle_line_raw.ljust(line_length)
    coupling_line = coupling_line_raw[4:-5].ljust(line_length)
    
    final_J_str = coupling_line_raw[-5:-1]
    final_double_J = J_to_doubleJ(final_J_str)
    
    subshell_List = chunk_string(subshells_line, 9)
    middle_line_List = chunk_string(middle_line, 9)
    coupling_line_List = chunk_string(coupling_line, 9)

    csf_descriptor = np.zeros(3 * len(peel_subshells_List), dtype=np.float32)
    orbs_occupied_indices = []
    
    for i, (subshell_charges, middle_line_item, coupling_line_item) in enumerate(zip(subshell_List, middle_line_List, coupling_line_List)):
        subshell = subshell_charges[:5].strip()
        subshell_electron_num = int(subshell_charges[6:8])
        is_last = (i == len(subshell_List) - 1)
        
        temp_middle_item = 0
        if not middle_line_item.isspace():
            temp_middle_item = J_to_doubleJ(middle_line_item.split(';')[-1].strip())
        
        temp_coupling_item = 0
        if not coupling_line_item.isspace():
            temp_coupling_item = J_to_doubleJ(coupling_line_item.strip())
        elif not middle_line_item.isspace():
            temp_coupling_item = temp_middle_item

        if is_last:
            temp_coupling_item = final_double_J

        try:
            orbs_index = peel_subshells_List.index(subshell)
            descriptor_index = orbs_index * 3
        except ValueError:
            continue
        
        orbs_occupied_indices.append(orbs_index)
        csf_descriptor[descriptor_index:descriptor_index+3] = [subshell_electron_num, temp_middle_item, temp_coupling_item]
    
    all_orbs_indices = set(range(len(peel_subshells_List)))
    occupied_orbs_indices = set(orbs_occupied_indices)
    remaining_orbs_indices = list(all_orbs_indices - occupied_orbs_indices)

    for index in remaining_orbs_indices:
        csf_descriptor[index*3 + 2] = final_double_J
        
    return csf_descriptor

def parse_csf_2_descriptor_with_subshell(peel_subshells_List: List[str], csf: List[str]) -> np.ndarray:
    """
    Parses a single CSF into a detailed descriptor array (5 features per orbital).
    Features: [main_quantum_num, kappa, electron_count, intermediate_J, coupled_J]
    """
    subshells_line, middle_line_raw, coupling_line_raw = [line.rstrip() for line in csf]
    line_length = len(subshells_line)
    middle_line = middle_line_raw.ljust(line_length)
    coupling_line = coupling_line_raw[4:-5].ljust(line_length)
    
    final_J_str = coupling_line_raw[-5:-1]
    final_double_J = J_to_doubleJ(final_J_str)
    
    subshell_List = chunk_string(subshells_line, 9)
    middle_line_List = chunk_string(middle_line, 9)
    coupling_line_List = chunk_string(coupling_line, 9)
    
    csf_descriptor = np.zeros(5 * len(peel_subshells_List), dtype=np.float32)
    orbs_occupied_indices = []
    
    for idx, subshell in enumerate(peel_subshells_List):
        main_quantum_num = int(''.join(filter(str.isdigit, subshell)))
        orbital_part = ''.join(filter(lambda x: not x.isdigit(), subshell))
        if not orbital_part.endswith(' ') and not orbital_part.endswith('-'):
            orbital_part += ' '
        kappa_value = str_subshell_2_kappa(orbital_part)
        descriptor_index = idx * 5
        csf_descriptor[descriptor_index] = main_quantum_num
        csf_descriptor[descriptor_index + 1] = kappa_value
    
    for i, (subshell_charges, middle_line_item, coupling_line_item) in enumerate(zip(subshell_List, middle_line_List, coupling_line_List)):
        subshell = subshell_charges[:5].strip()
        subshell_electron_num = int(subshell_charges[6:8])
        is_last = (i == len(subshell_List) - 1)
        
        is_full = if_subshell_full_charged(subshell, subshell_electron_num)
        
        temp_middle_item = 0
        if not middle_line_item.isspace():
            temp_middle_item = J_to_doubleJ(middle_line_item.split(';')[-1].strip())
            if not is_full: temp_middle_item *= 2
        
        temp_coupling_item = 0
        if not coupling_line_item.isspace():
            temp_coupling_item = J_to_doubleJ(coupling_line_item.strip())
            if not is_full: temp_coupling_item *= 2
        elif not middle_line_item.isspace():
            temp_coupling_item = temp_middle_item
        
        if is_last:
            temp_coupling_item = final_double_J * (2 if not is_full else 1)
        
        try:
            orbs_index = peel_subshells_List.index(subshell)
            descriptor_index = orbs_index * 5
        except ValueError:
            continue
        
        orbs_occupied_indices.append(orbs_index)
        
        if is_full:
            temp_middle_item = 0
            temp_coupling_item = 0
        
        csf_descriptor[descriptor_index + 2] = subshell_electron_num
        csf_descriptor[descriptor_index + 3] = temp_middle_item
        csf_descriptor[descriptor_index + 4] = temp_coupling_item
    
    all_orbs_indices = set(range(len(peel_subshells_List)))
    occupied_orbs_indices = set(orbs_occupied_indices)
    remaining_orbs_indices = list(all_orbs_indices - occupied_orbs_indices)
    
    for index in remaining_orbs_indices:
        csf_descriptor[index*5 + 4] = final_double_J * 2
        
    return csf_descriptor

# --- Worker Function for Multiprocessing ---
def _worker_parse_csf(args: Tuple[List[str], List[str], bool]) -> Optional[np.ndarray]:
    """
    Worker function to parse a single CSF. Designed for use with multiprocessing.Pool.
    """
    csf_item, peel_subshells_List, with_subshell_info = args
    try:
        if len(csf_item) != 3:
            return None
        
        if with_subshell_info:
            return parse_csf_2_descriptor_with_subshell(peel_subshells_List, csf_item)
        else:
            return parse_csf_2_descriptor(peel_subshells_List, csf_item)
    except Exception:
        return None

# --- Main Parallel Processing Functions ---

def batch_process_csfs_to_descriptors_parallel(
    CSFs_file_data: CSFs, 
    with_subshell_info: bool = False,
    num_cores: Optional[int] = None
) -> np.ndarray:
    """
    Batch process all CSFs in a file to descriptor arrays using multiple CPU cores.
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    print(f"Starting parallel CSF processing on {num_cores} cores...")
    
    peel_subshells_List = get_CSFs_peel_subshells(CSFs_file_data)
    all_csfs = [csf for block in CSFs_file_data.CSFs_block_data for csf in block]
    tasks = [(csf, peel_subshells_List, with_subshell_info) for csf in all_csfs]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(_worker_parse_csf, tasks), total=len(tasks), desc="Generating Descriptors"))

    all_descriptors = [res for res in results if res is not None]
    
    if not all_descriptors:
        raise ValueError("No valid CSF data could be processed!")
        
    descriptors_array = np.stack(all_descriptors)
    
    print(f"\nSuccessfully processed {len(descriptors_array)} CSFs in parallel.")
    print(f"Descriptor array shape: {descriptors_array.shape}")
    
    return descriptors_array

def batch_process_csfs_with_multi_block_parallel(
    CSFs_file_data: CSFs,
    label_type: str = 'block',
    with_subshell_info: bool = False,
    num_cores: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch process CSFs to descriptors and labels in parallel.
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    print(f"Starting parallel CSF processing with labels on {num_cores} cores...")
    
    peel_subshells_List = get_CSFs_peel_subshells(CSFs_file_data)
    
    tasks, labels = [], []
    global_csf_counter = 0
    for block_idx, block in enumerate(CSFs_file_data.CSFs_block_data):
        for csf_idx, csf_item in enumerate(block):
            tasks.append((csf_item, peel_subshells_List, with_subshell_info))
            if label_type == 'block':
                labels.append(block_idx)
            elif label_type == 'sequential':
                labels.append(csf_idx)
            elif label_type == 'global_sequential':
                labels.append(global_csf_counter)
            else:
                labels.append(f"block_{block_idx}_csf_{csf_idx}")
            global_csf_counter += 1

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(_worker_parse_csf, tasks), total=len(tasks), desc="Generating Descriptors with Labels"))

    valid_descriptors, valid_labels = [], []
    for descriptor, label in zip(results, labels):
        if descriptor is not None:
            valid_descriptors.append(descriptor)
            valid_labels.append(label)

    if not valid_descriptors:
        raise ValueError("No valid CSF data could be processed!")

    descriptors_array = np.stack(valid_descriptors)
    labels_array = np.array(valid_labels)

    print(f"\nSuccessfully processed {len(descriptors_array)} CSFs with labels in parallel.")
    print(f"Descriptor array shape: {descriptors_array.shape}")
    print(f"Labels array shape: {labels_array.shape}")

    return descriptors_array, labels_array

# --- Example Usage ---
if __name__ == '__main__':
    # This block demonstrates how to use the functions.
    # It creates a mock CSF data object and runs the parallel processing.
    
    print("--- Running Self-Contained Example ---")

    # 1. Define mock data for demonstration
    mock_subshell_info = [
        "Peel subshells:",
        "5s  4d-  4d  5p-  5p  6s  4f-  4f  5d"
    ]
    
    # A list containing two blocks of CSFs. Each CSF is a list of 3 strings.
    mock_csf_blocks = [
        [ # Block 0
            [
                "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)  4f-( 1)  4f ( 6)  5d ( 1)",
                "                                                                       3/2      ",
                "                                                                            4-  "
            ],
            [
                "  5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)  4f-( 2)  4f ( 5)  5d ( 1)",
                "                                                                       5/2      ",
                "                                                                            3+  "
            ]
        ],
        [ # Block 1
            [
                "  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)  4f-( 7)  5d-( 2)  5d ( 2)  6p-( 1)",
                "                                                                       1/2      ",
                "                                                                            2-  "
            ]
        ]
    ]

    # 2. Create the CSFs data object
    csf_data_object = CSFs(subshell_info_raw=mock_subshell_info, csfs_block_data=mock_csf_blocks)

    # 3. Run the parallel functions
    try:
        print("\n--- Testing batch_process_csfs_to_descriptors_parallel ---")
        # Use 2 cores for the example, or fewer if not available
        num_cores_to_use = min(2, multiprocessing.cpu_count())
        
        descriptors = batch_process_csfs_to_descriptors_parallel(
            csf_data_object, 
            with_subshell_info=False,
            num_cores=num_cores_to_use
        )
        print("Sample descriptor (first row):\n", descriptors[0])

        print("\n--- Testing batch_process_csfs_with_multi_block_parallel ---")
        X, y = batch_process_csfs_with_multi_block_parallel(
            csf_data_object,
            label_type='block',
            with_subshell_info=True,
            num_cores=num_cores_to_use
        )
        print("Sample detailed descriptor (first row):\n", X[0])
        print("Generated labels:\n", y)

    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")

    print("\n--- Example Finished ---")