MODULE=imagedata

all:	sdist test

test:
	tox

sdist:
	python3 -m pep517.build .

test_upload:
	twine upload --skip-existing --repository testpypi dist/*

upload:
	twine upload --skip-existing dist/*

git:
	#git tag -a $(call next_patch_ver)
	git push origin master --tags

html:
	cd docs; \
	sphinx-apidoc -o source/ ../; \
	make html

test_install:
	pip install --upgrade --index-url https://test.pypi.org/simple/ imagedata
