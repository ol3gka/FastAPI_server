# FastAPI_server

Dataset and baseline model architecture are taken from previous work (https://github.com/ol3gka/OTUS_Machine-Learning.-Advanced_2023)
A) Create virtual environment and activate it
1) `python -m venv env`
2) `./env/Scripts/activate`

B) Clone repo
`git clone https://github.com/ol3gka/FastAPI_server.git`
С) Run Fast API server
1) Build Docker image using `docker build . -t fastapi_ml_server_nikolaev`

2) Run Docker container using `docker run --rm -it -p 80:80 fastapi_ml_server_nikolaev`

3) Go to http://127.0.0.1:80/docs to see all available methods of the API

In pictures folder screens how server works are presented
