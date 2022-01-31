start:
	python ./bloodtype.py


env:
	python3 -m venv venv
	venv/bin/pip3 install numpy matplotlib numba pandas tqdm ipdb

rm_env:
	rm -rf ./venv
