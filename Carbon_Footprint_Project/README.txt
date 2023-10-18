DESCRIPTION


This code package contains five Jupyter Notebook scripts and one tableau workbook to show the final visualization product. It also contains three CSV files for running the main prophet model notebook and the cross-validation notebooks. We used Jupyter Notebook as it makes working with datasets and models easier. 


The file `prophet_model_final.ipynb` is used for predicting the CO₂ emissions from 2022 to 2026 for each of the US states. FaceBook(FB) Prophet model is used to do the predictions. For training the model we have historical data on CO₂ and various lifestyle variables from 1997 to 2021. 


The file `CV_on_combined_US_data.ipynb` is used for cross-validation of the model predictions for combined US data.


The file `CV_on_random_state.ipynb` is used for cross-validation of the model predictions for a random US state. 


The file `rawCSV_to_Tidy.ipynb` is used to compile data from various sources. 


The file `prepare_model_results_dataset_for_visualization.ipynb` is used for preparing and formatting the model result dataset for use in tableau visualization. This was required for a specific kind of line chart that we have in our visualization.


INSTALLATION


1. Install Anaconda following the steps from https://docs.anaconda.com/anaconda/install/index.html.
2. Create an environment with python=3.8 as prophet requires python version 3.8.
   * `conda create --name <an-env-name> python=3.8`
3. Activate the environment.
   * `Conda activate <env-name-created-above>`
4. Install FB Prophet using pip: 
   * `pip install prophet`
5. Install Jupyter using pip:
   * `pip install jupyter`




EXECUTION
For running `prophet_model_final.ipynb`:
1. Open Jupyter Notebook using the command: jupyter notebook.
2. Once the Jupyter Notebook open up in the browser, navigate to the directory where the file `prophet_model_final.ipynb` is located and open it up.
3. Ensure the correct CSV file path is provided to the ‘read_csv()’ function in the third cell.
4. To check out the model with the demo dataset, set the `is_demo` variable to `False` in the fourth cell.
5. In the menu, click the Kernel dropdown and select `Restart and run all`.
6. If there are no errors, a CSV file with the predicted CO₂ values for 2022 to 2026 will be generated with the name ‘predicted_values.csv’ on the exact location of the Jupyter Notebook. The historical values for CO₂ emissions and lifestyle variables are also included in the file.




For running  `CV_on_combined_US_data.ipynb` and CV_on_random_state.ipynb`:
1. Once the Jupyter Notebook opens up in the browser, navigate to the directory where the file is located and open it up.
2. Ensure the correct CSV file path is provided to the ‘read_csv()’ function.
3. In the menu, click the Kernel dropdown and select `Restart and run all`.
4. If there are no errors, you’ll see the cross-validation metrics at the end of the notebook.




For running `rawCSV_to_Tidy.ipynb`:
1. Download the three raw datasets from the sources mentioned in the file itself.
2. Update the CSV file paths in the second cell.
3. In the menu, click the Kernel dropdown and select `Restart and run all`.
4. If there are no errors, a CSV file should be generated in the same folder with the name `tidy_dataset2_v4.csv`.


For running `prepare_model_results_dataset_for_visualization.ipynb`:
1. Once the Jupyter Notebook opens up in the browser, navigate to the directory where the file is located and open it up.
2. Ensure the correct CSV file path is provided to the ‘read_csv()’ function. The CSV file is the file generated from the Prophet model using the file `prophet_model_final.ipynb`.
3. In the menu, click the Kernel dropdown and select `Restart and run all`.
4. If there are no errors, you should have a CSV generated in the same folder as the file. 
5. This file can then be used in tableau visualization. To get the tableau workbook of our project go to the following link and download it: shorturl.at/dLMU6 . You will need the Tableau application to open up the workbook. Once opened in the application, replace the CSV file used in the workbook with the one you generated.