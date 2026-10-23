# SafeNet KZ — негізгі командалар. Логиканың өзі tasks.py ішінде
# (Windows-та make жоқ болса, сол командаларды make.bat орындайды).

PY ?= python
N ?= 0
PORT ?= 8000

.PHONY: install tessdata data train eval bench docs api app test test-stage demo serve

install:
	$(PY) tasks.py install

tessdata:
	$(PY) tasks.py tessdata

data:
	$(PY) tasks.py data

train:
	$(PY) tasks.py train

eval:
	$(PY) tasks.py eval

bench:
	$(PY) tasks.py bench

docs:
	$(PY) tasks.py docs

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

serve:
	$(PY) tasks.py serve PORT=$(PORT)
