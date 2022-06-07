# code to build image
FROM python:3.8.13-buster

# COPY api /api
COPY skin_cancer_detection /skin_cancer_detection
# EDIT: need correct joblib
COPY model_resnet_224_augmented_ws.joblib /model_resnet_224_augmented_ws.joblib
COPY requirements.txt /requirements.txt
# EDIT: create predict.py or edit website.py to include required information and commands
# COPY predict.py /predict.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
