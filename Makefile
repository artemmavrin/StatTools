PYTHON := python3

.PHONY: install test clean

install:
	rm -rf build dist stattools.egg-info
	${PYTHON} setup.py install

test:
	${PYTHON} -m unittest discover --verbose

clean:
	find . -name "__pycache__" -type d | xargs rm -rf
	find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	find . -name "*.pyc" -type f | xargs rm -f
