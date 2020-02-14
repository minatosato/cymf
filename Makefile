all:
	python setup.py build_ext --inplace
install:
	python setup.py install
clean:
	rm ./cymf/*.cpp
	rm -rf ./build
	rm ./cymf/*.so
