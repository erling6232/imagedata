init:
	pip3 install -r requirements.txt

test:
	#nosetests3 -v -s tests
	nosetests3 tests
