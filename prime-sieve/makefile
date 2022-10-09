all: compile run clean

compile: main.cc
	mpic++ main.cc -std=c++20
run: ./a.out
	mpirun --hostfile hostfile -c 2 ./a.out 1000000
	mpirun --hostfile hostfile -c 3 ./a.out 1000000
	mpirun --hostfile hostfile -c 5 ./a.out 1000000
	mpirun --hostfile hostfile -c 9 ./a.out 1000000
	mpirun --hostfile hostfile -c 17 ./a.out 1000000
clean: ./a.out
	rm a.out
