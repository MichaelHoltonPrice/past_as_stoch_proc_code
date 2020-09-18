# Installation
## Option 2: Build the Docker image using the Dockerfile
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