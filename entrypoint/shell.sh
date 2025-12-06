cd /mvm/
export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate
python3 -m pip install -e .
exec bash
