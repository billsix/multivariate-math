export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate
cd /mvm
# Register the live bind-mounted tree as the editable package so the notebooks
# can import it, whether launched via `make jupyter` or from a shell (mirrors
# shell.sh).  Dependencies are already baked into the image's venv.
uv pip install --python $(which python) --no-deps --no-index --no-build-isolation -e .
# in general this is super dangerous, but for our purposes,
# it's fine
python -m ipykernel install --user --name=mvm
exec jupyter lab \
         --allow-root \
         --ip=0.0.0.0 \
         --port=8888 \
         --ServerApp.token='' \
         --ServerApp.password='' \
         --ServerApp.disable_check_xsrf=True \
         --no-browser \
         --MultiKernelManager.default_kernel_name=mvm
