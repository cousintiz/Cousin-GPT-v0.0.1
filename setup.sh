#! /usr/bin/bash

# file to setup development environment
# - create venv
# - activate venv (linux)
# - installs all libraries
# - enables the file for git and app


# env setup
python3.10 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

# permission for files
chmod u+x push.sh
chmod u+x app.sh


# WARNNING: MIGHT HAVE TO CHANGE python version (python3.<version>)