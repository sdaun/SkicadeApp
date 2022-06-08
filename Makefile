# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* skin-cancer-detection/*.py

black:
	@black scripts/* skin-cancer-detection/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr skin-cancer-detection-*.dist-info
	@rm -fr skin-cancer-detection.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#      UPLOAD TO GCP
# ----------------------------------

# project id - replace with your GCP project id
PROJECT_ID=eng-scene-346915

# bucket name - replace with your GCP bucket name
BUCKET_NAME=wagon-data-871-daun

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

# change local path to upload other files
LOCAL_PATH=~/code/sdaun/skin-cancer-detection/raw_data/HAM10000_all

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
	@gsutil cp -r ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}


# ----------------------------------
#      TRAINING ON GCP
# ----------------------------------

##### Machine configuration - - - - - - - - - - - - - - - -

# REGION=europe-west1

#PYTHON_VERSION=3.7
#FRAMEWORK=scikit-learn
#RUNTIME_VERSION=1.15

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=skin-cancer-detection
FILENAME=model

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -
# TO ADD
JOB_NAME=ADD_PIPELINE_OR_FUNCTION$(shell date +'%Y%m%d_%H%M%S')


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ __pycache__
	@rm -fr build dist *.dist-info *.egg-info
	@rm -fr */*.pyc

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

APP_NAME = skicade

streamlit:
	-@streamlit run website.py

heroku_login:
	-@heroku login

heroku_upload_public_key:
	-@heroku keys:add ~/.ssh/id_ed25519.pub

heroku_create_app:
	-@heroku create --ssh-git ${APP_NAME}

deploy_heroku:
	-@git push heroku main
	-@heroku ps:scale web=1

# ----------------------------------
