export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate
cd /mvm/
# All dependencies are baked into the image's venv at build time (see
# Dockerfile); this only registers the live bind-mounted tree as the editable
# package, so container start needs no downloads.
uv pip install --python $(which python) --no-deps --no-index --no-build-isolation -e .
exec bash
