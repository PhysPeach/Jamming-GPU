test: testsrc/test.o cu/MT.o cu/conf.o testcu/conf_test.o cu/particles.o testcu/particles_test.o cu/cells.o testcu/cells_test.o cu/jamming.o testcu/jamming_test.o
	nvcc -arch=sm_60 -o $@ $^

findJamming: main/findJamming.o cu/MT.o cu/conf.o cu/particles.o cu/cells.o cu/jamming.o
	nvcc -arch=sm_60 -o $@ $^

squeezeJamming: main/squeezeJamming.o cu/MT.o cu/conf.o cu/particles.o cu/cells.o cu/jamming.o
	nvcc -arch=sm_60 -o $@ $^

%.o: %.cu
	nvcc -arch=sm_60 -o $@ -c $<
