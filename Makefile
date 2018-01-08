PYTHON := python3

.PHONY: test clean

test:
	@- \
	for test in $$(find . -name "test_*.py" | sort | sed "s|^\./||"); \
	do \
	    echo "=== Running unit tests in '$$test'"; \
	    ${PYTHON} -m unittest $$test; \
	done;

clean:
	find . -name "__pycache__" -type d | xargs rm -rf
	find . -name ".ipynb_checkpoints" -type d | xargs rm -rf
	find . -name "*.pyc" -type f | xargs rm -f
