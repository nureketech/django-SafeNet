# SafeNet KZ — негізгі командалар. Логиканың өзі tasks.py ішінде
# (Windows-та make жоқ болса, сол командаларды make.bat орындайды).

PY ?= python
N ?= 0
PORT ?= 8000

.PHONY: install data train eval api app test test-stage demo

install:
	$(PY) tasks.py install

data:
	$(PY) tasks.py data

train:
	$(PY) tasks.py train

eval:
	$(PY) tasks.py eval

api:
	$(PY) tasks.py api PORT=$(PORT)

app:
	$(PY) tasks.py app PORT=$(PORT)

test:
	$(PY) tasks.py test

test-stage:
	$(PY) tasks.py test-stage N=$(N)

demo:
	$(PY) tasks.py demo PORT=$(PORT)
