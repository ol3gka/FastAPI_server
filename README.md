# FastAPI_server

Dataset and baseline model architecture are taken from previous work (https://github.com/ol3gka/OTUS_Machine-Learning.-Advanced_2023)


A) Clone repo
1) `git clone https://github.com/ol3gka/FastAPI_server.git`

B) Create virtual environment and activate it
1) cd FastAPI_server
2) `python -m venv env`
3) `./env/Scripts/activate`

С) Run Fast API server
1) Build Docker image using `docker build . -t fastapi_ml_server_nikolaev`

2) Run Docker container using `docker run --rm -it -p 80:80 fastapi_ml_server_nikolaev`

3) Go to http://127.0.0.1:80/docs to see all available methods of the API

or run `query_example.py` in your IDE (e.g. vs code)

or ping  server through Thunder Client (vs code): POST http://127.0.0.1:80/predict
with JSON from [json_post_query_example.txt](pictures/json_post_query_example.txt)

In pictures folder screens how server works are presented
