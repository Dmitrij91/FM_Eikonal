all:
	cd Fast_Marching_Cython; \
	python setup.py build_ext --inplace
	rm -f -r Fast_Marching_Cython/build
	rm -f Fast_Marching_Cython/Fast_*.c
	rm -f Fast_Marching_Cython/Distance*.c
clean:
	rm -f Fast_Marching_Cython/Fast_*.so
	rm -f Fast_Marching_Cython/Distance*.so
