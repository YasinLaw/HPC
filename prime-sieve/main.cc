#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <memory>

int main(int argc, char* argv[])
{
	auto timer_start = std::chrono::steady_clock::now();

	int rank, size;

	int64_t prime;

	if (argc != 2) return 1;
	int64_t length = atoi(argv[1]);
	int64_t bound = sqrt(length);
	int64_t sieve_start = bound + 1;
	int64_t range = length - bound;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (length % (size - 1) != 0) 
	{
		printf("input invalid, program terminated.\n");
		MPI_Finalize();
		return -1;
	}

	const int source = size - 1;

	int step = ceil(length / (double)(size - 1));
	int64_t left = sieve_start + step * rank;
	int64_t right = sieve_start + step * (rank + 1) > length ? length : sieve_start + step * (rank + 1);
	int64_t offset = left;
	int64_t count = 0;

	if (rank == source)
	{
		bool* array = new bool[bound + 1];
		memset(array, true, sizeof(bool) * length);
		for (int64_t i = 2; i <= bound; i++)
		{
			if (array[i]) 
			{
				prime = i;
				MPI_Bcast(&prime, 1, MPI_INT64_T, source, MPI_COMM_WORLD);
				for (int64_t j = pow(i, 2); j <= bound; j += i)
				{
					array[j] = false;
				}
			}
		}
		for (int64_t i = 2; i <= bound; i++) 
		{
			if (array[i]) count++;
		}
		prime = -1;
		MPI_Bcast(&prime, 1, MPI_INT64_T, source, MPI_COMM_WORLD);
		int64_t task_count = 0;
		for (int i = 0; i < source; i++) 
		{
			MPI_Recv(&task_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			count += task_count;
		}
		printf("for this round we used %d processors, got %lld prime numbers between 0 and %lld\n", size, count, length);
		delete [] array;
	}
	else
	{
		bool* task_array = new bool[right - left];
		memset(task_array, true, sizeof(bool) * (right - left));
		while (true)
		{
			MPI_Bcast(&prime, 1, MPI_INT64_T, source, MPI_COMM_WORLD);
			if (prime == -1) 
			{
				for (int64_t i = left; i < right; i++) 
				{
					if (task_array[i - offset]) count++;
				}
				MPI_Send(&count, 1, MPI_INT, source, 0, MPI_COMM_WORLD);
				delete [] task_array;
				MPI_Finalize();
				return 0;
			};

			int64_t start = ceil(left / (double) prime) * prime;
			for (int64_t i = start; i < right; i += prime)
			{
				task_array[i - offset] = false;
			}
		}
	}

	MPI_Finalize();

	auto timer_end = std::chrono::steady_clock::now();
	auto elapsed = timer_end - timer_start;
	printf("and we cost %lld μs\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
	return 0;
}
