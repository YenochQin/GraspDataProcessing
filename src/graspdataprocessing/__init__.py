#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :__init__.py
@date :2025/06/16 15:59:13
@author :YenochQin (秦毅)
'''

__author__ = "YenochQin (秦毅)"

from .version import __version__

from .data_IO import (
    GraspFileLoad,
    EnergyFile2csv,
    write_sorted_CSFs_to_cfile,
    save_csf_metadata,
    save_csfs_binary,
    continue_calculate,
    update_config,
    csfs_index_storange,
    precompute_large_hash,
    save_descriptors,
    save_descriptors_with_multi_block,
    load_csf_metadata,
    load_csfs_binary,
    csfs_index_load,
    load_large_hash,
    load_config,
    load_descriptors,
    load_descriptors_with_multi_block
)

from .utils import (
    MixCoefficientData,
    CSFs,
    level_print_title,
    level_J_value,
    level_parity,
    energy_au_cm,
    align_2d_list_columns,
    int_nl_2_str_nl,
    str_subshell_2_kappa,
    doubleJ_to_J,
    lsj_transition_data_level_location,
    transition_data_level_location,
    transition_dT_cal,
    read_fortran_record,
    chunk_string,
    level_data_compare
)

from .CSFs_choosing import (
    batch_asfs_mix_square_above_threshold,
    asf_mix_square_above_threshold_coupling_info,
    CSFs_block_get_CSF,
    batch_blocks_csfs_final_coupling_J_collection,
    single_asf_csfs_final_coupling_J_mix_coefficient_sum,
    single_block_batch_asfs_CSFs_final_coupling_J_collection,
    batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum,
    block_csfs_coupling_J_chosen,
    union_lists_with_order,
    merge_multiple_dicts_with_ordered_union,
    merge_csfs_indices_lists_by_block_key,
    CSFs_sort_by_mix_coefficient,
    generate_unique_random_numbers,
    radom_choose_csfs,
    process_block,
    maping_two_csfs_indices,
    csf_J,
    J_to_doubleJ,
    CSF_info_2_dict,
    CSF_item_2_dict,
    get_CSFs_file_info,
    
    batch_process_csfs_to_descriptors,
    batch_process_csfs_with_multi_block,
    create_csf_dataset_for_ml
)

from .processing import (
    ConfigurationFormat,
    LevelsEnergyData,
    mcdhf_energy_data_collection,
    ci_energy_data_collection,
    LevelsASFComposition,
    asf_radial_wavefunction_collection,
    RadialElectrondensityFunction,
    TransitionDataCollection,
    LSJTransitionDataCollection,
    LSJTransitionDataBlock,
    TransitionDataBlock,
    data_process
)

from .machine_learning_module import (
    ANNClassifier,
    setup_logging,
    setup_directories,
    initialize_iteration_results_csv,
    validate_initial_files,
    load_data_files,
    check_configuration_coupling,
    generate_chosen_csfs_descriptors,
    get_stay_descriptors,
    train_model,
    evaluate_model,
    save_iteration_results,
    check_grasp_cal_convergence,
    handle_calculation_error,
    get_unselected_descriptors,
    save_and_plot_results
)

__all__ = [
    # 版本信息
    '__author__',
    '__version__',
    
    # data_IO
    ## grasp_raw_data_load
    'GraspFileLoad',
    'EnergyFile2csv',
    ## produced_data_write
    'write_sorted_CSFs_to_cfile',
    'save_csf_metadata',
    'save_csfs_binary',
    'continue_calculate',
    'update_config',
    'csfs_index_storange',
    'precompute_large_hash',
    'save_descriptors',
    'save_descriptors_with_multi_block',
    ## processing_data_load
    'load_csf_metadata',
    'load_csfs_binary',
    'csfs_index_load',
    'load_large_hash',
    'load_config',
    'load_descriptors',
    'load_descriptors_with_multi_block',
    
    # utils
    'MixCoefficientData',
    'CSFs',
    # 工具函数
    'level_print_title',
    'level_J_value',
    'level_parity',
    'energy_au_cm',
    'align_2d_list_columns',
    'int_nl_2_str_nl',
    'str_subshell_2_kappa',
    'doubleJ_to_J',
    'lsj_transition_data_level_location',
    'transition_data_level_location',
    'transition_dT_cal',
    'read_fortran_record',
    'chunk_string',
    'level_data_compare',

    # CSFs_choosing
    ## CSFs_choosing
    'batch_asfs_mix_square_above_threshold',
    'asf_mix_square_above_threshold_coupling_info',
    'CSFs_block_get_CSF',
    'batch_blocks_csfs_final_coupling_J_collection',
    'single_asf_csfs_final_coupling_J_mix_coefficient_sum',
    'single_block_batch_asfs_CSFs_final_coupling_J_collection',
    'batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum',
    'block_csfs_coupling_J_chosen',
    'union_lists_with_order',
    'merge_multiple_dicts_with_ordered_union',
    'merge_csfs_indices_lists_by_block_key',
    'CSFs_sort_by_mix_coefficient',
    'generate_unique_random_numbers',
    'radom_choose_csfs',
    'process_block',
    'maping_two_csfs_indices',
    
    ## CSFs_compress_extract
    'csf_J',
    'J_to_doubleJ',
    'CSF_info_2_dict',
    'CSF_item_2_dict',
    'get_CSFs_file_info',
    'batch_process_csfs_to_descriptors',
    'batch_process_csfs_with_multi_block',
    'create_csf_dataset_for_ml',
    
    # processing
    'ConfigurationFormat',
    'LevelsEnergyData',
    'mcdhf_energy_data_collection',
    'ci_energy_data_collection',
    'LevelsASFComposition',
    'asf_radial_wavefunction_collection',
    'RadialElectrondensityFunction',
    'TransitionDataCollection',
    'LSJTransitionDataCollection',
    'LSJTransitionDataBlock',
    'TransitionDataBlock',
    'data_process',
    
    # machine_learning_module
    ## ANN
    'ANNClassifier',
    ## machine_learning_initialization
    'setup_logging',
    'setup_directories',
    'initialize_iteration_results_csv',
    'validate_initial_files',
    'load_data_files',
    'check_configuration_coupling',
    'generate_chosen_csfs_descriptors',
    'get_stay_descriptors',
    ## machine_learning_training
    'train_model',
    'evaluate_model',
    'save_iteration_results',
    'check_grasp_cal_convergence',
    'handle_calculation_error',
    'get_unselected_descriptors',
    'save_and_plot_results'
]