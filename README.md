# Software Refactoring Prediction Model 🔄

Welcome to the Software Refactoring Prediction Model repository! This project utilizes machine learning techniques to predict code refactoring needs. It's designed to help developers maintain clean and efficient codebases by whether the developer need to refactor that particular piece of code or not.

## Introduction 🌟
  ### This project is part of a replication of an existing study 📄 
  - [Original Research Paper](https://arxiv.org/pdf/2001.03338)
  - [Old Codebase](https://github.com/refactoring-ai/predicting-refactoring-ml)
  - Their code used a software called Refactroing Miner, which is a task in itself to get it up and running. Contrary to that, we used the SQL Scripts provided to extract data in almost half the time.
    - [Data Fetching Scripts](https://zenodo.org/records/3547639)
  This project uses Python to analyze codebases and predict refactoring opportunities. It leverages several machine learning models to assess various aspects of the code and suggests potential refactoring to enhance code quality.
  
  ### Our Work:
  - Improved the code by remvoing the unwanted methods and features that weren't needed in the final project.
  - Included Randomized and Grid Search Cross Validation support.
  - Due to challenges in obtaining the optimal hardware for running this project, performed the same analysis and got similar results on a fraction ***(0.2%, 0.5% & 1.0%)*** of the original dataset.

## Getting Started 🚀
Follow these simple steps to get a local copy up and running.

## Prerequisites 📋
- Python 3.8+
- pip

## Installation 💽
1. Clone the repo:
   ```sh
   git clone https://github.com/Hetav01/Software-Refactoring-Prediction-Model.git

2. Extract the amount of dataset required for the pipeline from the [Data Fetching Scripts](https://zenodo.org/records/3547639).

3. Copy the CSV dataset in the `dataset` folder.

4. Edit the pathnames at required places, namely, `preprocessing/preprocessing.py`, `binaryClassification.py`, `testing/Runner_Test.py` and `testing/binaryClassification2.py`.

5. Before running the driver file for the entire pipeline, install all the required dependencies:
   ```sh
   pip3 install --user -r requirements.txt

6. The driver file for the code is either `binaryClassification.py` or `testing/binaryClassification2.py` depending on whether you want to just get the results or additionally test the models on unseen data(use `testing/binaryClassification2.py` for that).
   You can run either by executing the following command:
    ```sh
     python3 binaryClassification.py
    ``` 
    ```sh
     python3 testing/binaryClassification2.py
    ```   

The script will follow the configurations in the `configs.py`. There, you can define which datasets to analyze, which models to build, which under sampling algorithms to use, and etc. Please, read the comments of this file carefully.

7. For collecting the results, the Python scripts will automatically update the `result.txt` and `result_unseen.txt` files to provide you with the latest metrics. Refer to the terminal while the program is running to understand which Hyperparamters work best for each model.

## Contributing 🤝
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## License
This project is licensed under the MIT license.
