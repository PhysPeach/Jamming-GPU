test: testsrc/test.o cpp/MT.o cpp/conf.o testcpp/conf_test.o cpp/particles.o testcpp/particles_test.o cpp/cells.o testcpp/cells_test.o cpp/jamming.o testcpp/jamming_test.o
	g++ -o $@ $^

findJamming: main/findJamming.o cpp/MT.o cpp/conf.o cpp/particles.o cpp/cells.o cpp/jamming.o
	g++ -o $@ $^

squeezeJamming: main/squeezeJamming.o cpp/MT.o cpp/conf.o cpp/particles.o cpp/cells.o cpp/jamming.o
	g++ -o $@ $^

%.o: %.cpp
	g++ -o $@ -c $<
