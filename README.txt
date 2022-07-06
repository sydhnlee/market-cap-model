Predictive Analysis of the Stock Market Project

Overview
-----------
The project code consists of two notebooks, "Data Cleaning" and "Data Analysis". The "Data Cleaning" notebook is the code used to clean the raw data
retrieved from https://www.sec.gov/dera/data/financial-statement-data-sets.html and produces "clean_data.csv" which we use for all of our models in the "Data Analysis" notebook. In order for "Data Analysis" to run properly only "clean_data.csv" is required to be in the project directory, and is already included. **IMPORTANT** For "Data Cleaning" to run properly, the raw data files are required. These files have been excluded due to their large size but can be installed by following the instructions below.

Dependencies
------------
Data Cleaning:
yfinance

Data Analysis:
matplotlib
seaborn

Both:
pandas
numpy

Installation (Necessary to run "Data Cleaning" notebook)
--------------------------------------------------------
Note: The raw data found at https://www.sec.gov/dera/data/financial-statement-data-sets.html is included in the "raw_data.zip"

1. Install raw_data.zip and uncompress the file (**WARNING** ~16GB Uncompressed)
2. Inside the uncompressed raw_data file should be a "data" directory
3. Place the "data" directory inside of the main project directory (the same directory this readme file is in) 
4. You are now ready to execute the "Data Cleaning" notebook

Execution
---------
Assuming the listed dependencies are installed and the installation steps are followed, both notebooks should execute properly when running all cells

Acknowledgements
----------------
https://github.com/ranaroussi/yfinance
