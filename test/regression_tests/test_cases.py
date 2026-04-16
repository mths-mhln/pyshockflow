import subprocess
from pathlib import Path
import pytest
import os
import sys
import shutil
import pickle
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_DIR = BASE_DIR / "regression_tests"

def discover_cases():
    cases = []
    for d in CASES_DIR.iterdir():
        if d.is_dir() and (d / "input.ini").exists():
            cases.append(d)
    
    print(f"Discovered {len(cases)} test cases:")
    for case in cases:
        print(f" - {case.name}")
    return cases


def run_solver(case_dir):
    log_file = case_dir / "log.txt"

    with open(log_file, "w") as f:
        try:
            result = subprocess.run(
                [sys.executable, "main.py"],
                cwd=case_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"Solver timeout in {case_dir}")

    assert result.returncode == 0, f"Solver failed with return code {result.returncode} in {case_dir}. Check log file: {log_file}"


def find_result_file(case_dir):
    results_dir = case_dir / "Results"
    assert results_dir.exists(), f"Results directory {results_dir} does not exist."
    assert results_dir.is_dir(), f"{results_dir} is not a directory."
    
    # Find subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    assert len(subdirs) == 1, (
        f"Expected exactly 1 subdirectory in {results_dir}, found {len(subdirs)}: {subdirs}"
    )

    result_subdir = subdirs[0]

    # Find .pik files
    pik_files = list(result_subdir.glob("*.pik"))
    assert len(pik_files) == 1, (
        f"Expected exactly 1 .pik file in {result_subdir}, found {len(pik_files)}: {pik_files}"
    )

    return pik_files[0]

def result_arrays_agree(reference_data, result_data, tol=1e-3):
    err = np.linalg.norm(reference_data - result_data) / np.linalg.norm(reference_data)
    if err<tol:
        return True, err
    else:
        return False, err

def compare_results_with_reference(case_dir):
    print(f"Comparing results for case: {case_dir.name}")
    
    reference_file = case_dir / "reference_result.pik"
    print(f"Reference file: {reference_file}")
    
    result_file = find_result_file(case_dir)
    print(f"Result file: {result_file}")
    
    file_paths = [reference_file, result_file]
    file_datas = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            file_datas.append(pickle.load(f))
    
    assert file_datas[0]['X Coords'].shape == file_datas[1]['X Coords'].shape, "X Coords shape mismatch between reference and result"
    
    keys_to_check = ['Pressure', 'Velocity', 'Density']
    for key in keys_to_check:
        they_agree, err = result_arrays_agree(file_datas[0]['Primitive'][key], file_datas[1]['Primitive'][key])
        assert they_agree, f"{key} field mismatch between reference and result with relative error {err:.2e}"
    



def clean_results(case_dir):
    results_dir = case_dir / "Results"
    if results_dir.exists():
        shutil.rmtree(results_dir)



@pytest.mark.parametrize("case_dir", discover_cases())
def test_run_case(case_dir):
    print(f"\nRunning test case: {case_dir.name}")

    clean_results(case_dir)
    run_solver(case_dir)
    compare_results_with_reference(case_dir)
    clean_results(case_dir)