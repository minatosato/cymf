all:
	python setup.py build_ext --inplace
clean:
	rm *.so
	rm -rf ./build
	rm *.cpp
