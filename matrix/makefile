all: compile run clean

compile: main.cc
	mpic++ main.cc -std=c++20
run: ./a.out
	mpirun --hostfile hostfile -c 1 ./a.out
	mpirun --hostfile hostfile -c 2 ./a.out
	mpirun --hostfile hostfile -c 4 ./a.out
	mpirun --hostfile hostfile -c 8 ./a.out
	# mpirun --hostfile hostfile -c 16 ./a.out
clean: ./a.out
	rm a.out
