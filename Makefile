all:
	python setup.py build_ext --inplace
install:
	python setup.py install
clean:
	rm ./fastmf/*.cpp
	rm -rf ./build
	rm ./fastmf/*.so
