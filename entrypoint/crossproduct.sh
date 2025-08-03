cd /mvm/
python3 -m pip install -e . --break-system-packages --root-user-action=ignore
python3 -m pip install --break-system-packages imgui
cd src/crossproduct/
exec python3 crossproduct.py

