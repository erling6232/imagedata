MODULE=imagedata

all:	sdist test

test:
	tox

sdist:
	python3 -m pep517.build .

test_upload:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

git:
	#git tag -a $(call next_patch_ver)
	git push origin master --tags
