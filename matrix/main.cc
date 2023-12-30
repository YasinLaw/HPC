#include <chrono>
#include <cstdio>
#include <random>

#include "mpi.h"

constexpr int len = 1024;

int main(int argc, char *argv[]) {
  int rank, size;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int matrix_size = len * len;
  const int chunk_size = matrix_size / size;

  double *a;
  double *b = new double[matrix_size];
  double *chunk = new double[chunk_size];
  double *results;
  double result;
  double *chunk_results = new double[chunk_size];
  auto start = std::chrono::steady_clock::now();

  if (rank == 0) {
    // initialize the matrix with random floating point number
    a = new double[matrix_size];
    results = new double[matrix_size];
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < len; i++) {
      for (int j = 0; j < len; j++) {
        a[i * len + j] = dist(mt);
        b[j * len + i] = dist(mt);
      }
    }
    start = std::chrono::steady_clock::now();
  }

  MPI_Scatter(a, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  MPI_Bcast(b, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int width = len;
  int height = len / size;

  int index = 0;
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      result = 0;
      for (int k = 0; k < len; k++) {
        result += chunk[r * len + k] * b[k * len + c];
      }
      chunk_results[index++] = result;
    }
  }

  MPI_Gather(chunk_results, chunk_size, MPI_DOUBLE, results + rank * chunk_size,
             chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;
    printf(
        "task costs %lld μs\n",
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    delete[] a;
    delete[] results;
  }
  delete[] b;
  delete[] chunk;
  delete[] chunk_results;

  MPI_Finalize();
  return 0;
}
