#include "mpi.h"
#include <cstdio>
#include <random>
#include <chrono>

constexpr int len = 1024;

int main(int argc, char* argv[])
{
	int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const int matrix_size = len * len;
	const int chunk_size = matrix_size / size;

	double* m1;
	double* m2 = new double[matrix_size];
	double* chunk = new double[chunk_size];
	double* r;
	double result;
	double* results = new double[chunk_size];
	auto start = std::chrono::steady_clock::now();

	if (rank == 0) 
	{
		// initialize the matrix with random floating point number
		m1 = new double[matrix_size];
		r = new double[matrix_size];
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (int i = 0; i < len; i++) 
		{
			for (int j = 0; j < len; j++)
			{
				m1[i * len + j] = dist(mt);
				m2[j * len + i] = dist(mt);
			}
		}
		start = std::chrono::steady_clock::now();
	}

	MPI_Scatter(m1, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(m2, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int width = len;
	int height = len / size;

	int index = 0;
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			result = 0;
			for (int k = 0; k < len; k++)
			{
				result += chunk[r * len + k] * m2[k * len + c];	
			}
			results[index++] = result;
		}
	}

	MPI_Gather(results, chunk_size, MPI_DOUBLE, r + rank * chunk_size, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		auto end = std::chrono::steady_clock::now();
		auto elapsed = end - start;
		printf("task costs %lld μs\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
		delete [] m1;
		delete [] r;
	}
	delete [] m2;
	delete [] chunk;
	delete [] results;

	MPI_Finalize();
	return 0;
}
