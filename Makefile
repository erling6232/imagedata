MODULE=imagedata

test:
	#nosetests3 -v -s tests
	nosetests3 tests

dist:
	python3 -m pep517.build .

# remove optional 'v' and trailing hash "v1.0-N-HASH" -> "v1.0-N"
git_describe_ver = $(shell git describe --tags | sed -E -e 's/^v//' -e 's/(.*)-.*/\1/')
git_tag_ver      = $(shell git describe --abbrev=0)
next_patch_ver = $(shell python versionbump.py --patch $(call git_tag_ver))
next_minor_ver = $(shell python versionbump.py --minor $(call git_tag_ver))
next_major_ver = $(shell python versionbump.py --major $(call git_tag_ver))

.PHONY: ${MODULE}/_version.py
${MODULE}/_version.py:
	echo '__version__ = "$(call git_describe_ver)"' > $@

.PHONY: release
#release: test lint mypy
release:
	#git tag -a $(call next_patch_ver)
	$(MAKE) ${MODULE}/_version.py
	#python setup.py check sdist upload # (legacy "upload" method)
	# twine upload dist/*  (preferred method)
	#git push origin master --tags
