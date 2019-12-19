# Setup virtual environment for model training
pip3 install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

echo " "
echo "Run ´source .venv/bin/activate´ to activate environment."
