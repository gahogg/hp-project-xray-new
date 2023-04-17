1. Download and unzip (or clone) the project wherever you choose (<YOUR_PATH>).

2. Enter terminal into <YOUR_PATH> and cd into <YOUR_PATH>/project/ folder.

3. Activate conda base env with `conda activate base`.

4. Create conda environment with `conda env create -f environment.yml`.

5. Activate the new environment with `conda activate hp_covid_env_new`.

6. Add opencv library to environment with `pip install opencv-python-headless`.

7. Run the app with `python main.py`.

The app should spin up a local server and give a link to view the app in a browser. 

If it fails, try repeating step 4 with `conda env create -f environment.yml python=3.9`, explicity specifying the python version. You may want to change that python version number, although it has been tested on 3.9.

To delete the conda environment, use `conda env remove -n hp_covid_env_new`


