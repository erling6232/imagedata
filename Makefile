MODULE=imagedata

all:	sdist test

test:
	tox

sdist:
	# python3 -m pep517.build .
	python3 -m build --sdist --wheel .

test_upload:
	twine upload --skip-existing --repository testpypi dist/*

upload:
	twine upload --skip-existing dist/*

git:
	git log --oneline --decorate
	git tag -a `cat VERSION.txt`
	git push origin master --tags

html:
	cd docs; \
	make clean; \
	sphinx-apidoc -o source/ ../; \
	make html

test_install:
	pip install --upgrade --index-url https://test.pypi.org/simple/ imagedata
