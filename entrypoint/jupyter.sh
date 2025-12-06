cd /mvm/
export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate
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
