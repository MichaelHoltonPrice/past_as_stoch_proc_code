# Installing requirements
TBD
## Option 1: Build the Docker image using the Dockerfile
First, clone the database (only the Dockerfile is in fact needed, and could instead be downloaded directly using the github web interface; the git protocol does not support directly downloading individual files).

```bash
git clone https://github.com/MichaelHoltonPrice/past_as_stoch_proc_code
cd past_as_stoch_proc_code
docker build -t michaelholtonprice/seshat .
```
```bash
docker run --name seshat -it michaelholtonprice/seshat
```




```bash
git clone https://github.com/MichaelHoltonPrice/past_as_stoch_proc_code
cd past_as_stoch_proc_code
docker build -t michaelholtonprice/seshat .
docker run --name seshat -it michaelholtonprice/seshat
```

The preceding code will build a new Docker image using the Dockerfile in the past_as_stoch_proc_code github repository and create a Docker container named seshat with the code placed in the /code directory. The user starts in the /code directory when the container is entered. To run the analysis use:

```bash
python3 seshat.py
```

## Option 2: Use a Docker image from Docker hub

## Option 3: Run on an existing machine. R 3.5 or later is needed. Details will vary based on the machine and operating system. The following should work with Ubuntu.

Clone the repository:
```bash
git clone https://github.com/MichaelHoltonPrice/past_as_stoch_proc_code
```

If necessary, install R (>=3.5.0). The following shell script can be used. It must be given execution permissions:
```bash
cd past_as_stoch_proc_code
sudo chmod +x ./install_R.sh
sudo ./install_R.sh
```

Install R dependencies:
```bash
sudo Rscript install_R_dependencies.R
```

Run the moralizing gods R code to generate data for Figure S1. See below for a summary of the code.
```bash
cd mhg_code
Rscript '!MoralizingGods.R'
```

Check the data against those used in the 2019 Nature paper by Whitehouse et al.:
```bash
cd mhg_code
Rscript check_moralising_gods_status.R
```

The data are not identical for the DoctrinalMode variable for to NGAs, Middle Yellow River Valley and Orkhon Valley. It's unclear what causes this discrepancy. However, only the MoralisingGods data are (not the Doctrinal Mode data) are used in our article.

Install dependencies, then run the python script that makes the publication figures. It is assumed that python3 and pip3 are already installed on the system.
```bash
cd ..
pip3 install -r requirements.txt
python3 make_figures.py
```

## Notes on the moralizing gods code
I (Michael Holton Price / MHP) started with the code in the following repository:

https://github.com/pesavage/moralizing-gods

However, the code in that repository uses hard-coded home directories of the authors and writes files to disk that are subsequently used in ways that make the main script, !MoralizingGods.R, run differently when called a second time. I therefore added minimal edits to make the code work. I have marked these edits with my initials, MHP. The exact commit of the preceding github repository that was used is bb63a09218ffb4e1723fa3a6e3da30baa0571cb1. While this is no longer the latest commit, using this earlier commit is fine because the results of the code I have provided match those in Whitehouse et al (2019).

The main script is !MoralizingGods.R. In the original github repository, /pesavage/moralizing-gods there are some starting .csv data files that are modified as the main script runs. Hence, it can only be run reproducibly once, and the starting .csv files must be replaced before re-running. The script clean_csv.R, which I have added to !MoralizingGods.R, handles this problem by deleting all .csv files in the mhg_code directory and copying in the original files from the direcotyr mhg_code/starting_csv.


In order to run the script with unmodified, starting .csv files
