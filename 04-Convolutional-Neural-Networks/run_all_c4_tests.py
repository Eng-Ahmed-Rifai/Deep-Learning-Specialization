"""
Course 4: Convolutional Neural Networks - Master Test Suite Runner
Deep Learning Specialization - Coursera
Automated Verification Suite by Eng. Ahmed Rifai
"""

import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TEST_SCRIPTS = [
    ("Week 2 Lab 1: ResNets", BASE_DIR / "Week-2-Deep-Convolutional-Models" / "ResNets" / "resnets.py"),
    ("Week 2 Lab 2: Transfer Learning", BASE_DIR / "Week-2-Deep-Convolutional-Models" / "Transfer-Learning" / "transfer_learning.py"),
    ("Week 3 Lab 1: YOLO Car Detection", BASE_DIR / "Week-3-Object-Detection" / "YOLO-Car-Detection" / "yolo_car_detection.py"),
    ("Week 3 Lab 2: Image Segmentation (U-Net)", BASE_DIR / "Week-3-Object-Detection" / "Image-Segmentation-UNet" / "unet.py"),
    ("Week 4 Lab 1: Face Recognition", BASE_DIR / "Week-4-Special-Applications" / "Face-Recognition" / "face_recognition.py"),
    ("Week 4 Lab 2: Neural Style Transfer", BASE_DIR / "Week-4-Special-Applications" / "Neural-Style-Transfer" / "neural_style_transfer.py"),
]


def run_all_tests():
    print("=" * 80)
    print("DEEP LEARNING SPECIALIZATION - COURSE 4 (CNN) VERIFICATION SUITE")
    print(f"Python Executable: {sys.executable}")
    print("=" * 80)
    
    results = []
    total_start = time.time()
    
    for idx, (name, script_path) in enumerate(TEST_SCRIPTS, 1):
        print(f"\n[{idx}/6] Executing: {name} ...")
        assert script_path.exists(), f"Target script does not exist: {script_path}"
        
        start = time.time()
        res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        elapsed = time.time() - start
        
        status = "PASSED" if res.returncode == 0 else "FAILED"
        results.append((name, status, elapsed, res.stdout, res.stderr))
        
        print(f"Result: {status} in {elapsed:.2f}s")
        if res.returncode != 0:
            print("STDERR Output:")
            print(res.stderr)
            
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    all_passed = True
    for name, status, elapsed, _, _ in results:
        print(f" - {name:<45}: [{status}] ({elapsed:.2f}s)")
        if status != "PASSED":
            all_passed = False
            
    total_elapsed = time.time() - total_start
    print("-" * 80)
    print(f"Total Test Time: {total_elapsed:.2f}s")
    if all_passed:
        print(">>> ALL 6 PROGRAMMING ASSIGNMENTS VERIFIED WITH 100% PASS RATE! <<<")
    else:
        print(">>> SOME TESTS FAILED. CHECK LOGS ABOVE. <<<")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
