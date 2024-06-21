import os
import subprocess

def run_get_crypto_data_script():
    print("Running getCryptoData.py...")
    subprocess.run(["python", "getCryptoData.py"])

def run_crypto_prediction_script():
    print("Running dataPredict.py...")
    subprocess.run(["python", "dataPredict.py"])

def main():
    run_get_crypto_data_script()
    run_crypto_prediction_script()

if __name__ == "__main__":
    main()
