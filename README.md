# Installing software requirements
## Option 1: Build the Docker image using the Dockerfile
First, clone the database and build the Docker image. Only the Dockerfile is need to build the image, and could instead be downloaded directly using the github web interface; the git protocol does not support directly downloading individual files.

```bash
git clone https://github.com/MichaelHoltonPrice/past_as_stoch_proc_code
cd past_as_stoch_proc_code
docker build -t michaelholtonprice/past_stoch_proc .
```

Start a container. Use the -v tag to mirror the directory to the right of the semicolon on the host machine (...past_stoch_proc) with the directory past_as_stoch_proc_code in the Docker container. This allows the results to be accessed on the host machine.

```bash
docker run --name past_stoch_proc -itv //c/Users/mpatm/past_stoch_proc:/past_as_stoch_proc_code
```

If, for whatever reason, it is not necessary to mirror the directory, use the following command:

```bash
docker run --name past_stoch_proc -it michaelholtonprice/past_stoch_proc
```

## Option 2: Use a Docker image from Docker hub

## Option 3: Run on an existing machine. R 3.5 or later is needed. Details will vary based on the machine and operating system. The following should work with Ubuntu or another Debian Linux flavor that uses the APT (the Advanced Package Protocol).

Clone the repository and enter the newly created directory:
```bash
git clone https://github.com/MichaelHoltonPrice/past_as_stoch_proc_code
cd past_as_stoch_proc_code
```

If necessary, install R (>=3.5.0). The following shell script can be used. It must be given execution permissions:
```bash
sudo chmod +x ./install_R.sh
sudo ./install_R.sh
```

Install R dependencies:
```bash
sudo Rscript install_R_dependencies.R
```

Install the Python dependencies. It is assumed that python3 and pip3 are already installed on the system.

```bash
cd ..
pip3 install -r requirements.txt
```

# Running the code
The following steps assume the user starts in the past_as_stoch_proc_code directory, and the steps to run the code are identical regardless of which option is used to satisfy the software requirements.

Enter the mhg_code directory and run the main analysis script to generate data for Figure S1. See below for a summary of the moralizing gods code.

```bash
cd mhg_code
Rscript '!MoralizingGods.R'
```

Check the data against those used in the 2019 Nature paper by Whitehouse et al.:
```bash
Rscript check_moralising_gods_status.R
```

The data are not identical for the DoctrinalMode variable for to NGAs, Middle Yellow River Valley and Orkhon Valley. It's unclear what causes this discrepancy. However, only the MoralisingGods data are used in our article (the Doctrinal Mode data are not used).

Return to the past_as_stoch_proc_code directory and run the Python script to make the publication figures.

```bash
cd ..
python3 make_figures.py
```

# Notes on the moralizing gods code
I (Michael Holton Price / MHP) started with the code in the following repository:

https://github.com/pesavage/moralizing-gods

However, the code in that repository uses hard-coded home directories of the authors and writes files to disk that are subsequently used in ways that make the main script, !MoralizingGods.R, run differently when called a second time. I therefore added minimal edits to make the code work. I have marked these edits with my initials, MHP. The exact commit of the preceding github repository that was used is bb63a09218ffb4e1723fa3a6e3da30baa0571cb1. While this is no longer the latest commit, using this earlier commit is fine because the results of the code I have provided match those in Whitehouse et al (2019).

The main script is !MoralizingGods.R. In the original github repository, /pesavage/moralizing-gods there are some starting .csv data files that are modified as the main script runs. Hence, it can only be run reproducibly once, and the starting .csv files must be replaced before re-running. The script clean_csv.R, which I have added to !MoralizingGods.R, handles this problem by deleting all .csv files in the mhg_code directory and copying in the original files from the directory mhg_code/starting_csv.

