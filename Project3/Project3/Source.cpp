#include <stdio.h>
#include <iostream>
#include <random>
#include "omp.h"

#define Q 5

#define LESS 0
#define EQUALS 1
#define GREATER 2

#define DEFAULT_SIZE 100000
#define DEFAULT_SELECTED 50000

/*Prints the elements of an array in a row. 
Used for debugging.*/
void print_array(double* arr, int n)
{
	for (int i = 0; i < n; i++)
	{
		std::cout << arr[i] << " ";
	}

	std::cout << std::endl;
}

/*Creates an array of size n with random elements in [0,100) 
rounded to one decimal digit.*/
void create_array(double* arr, int n)
{
	std::uniform_real_distribution<double> unif(0, 100);
	std::default_random_engine re;

	for (int i = 0; i < n; i++)
		arr[i] = std::round(10 * unif(re)) / 10;
}

/*Merges two sorted arrays. 
Used by sort.*/
void merge(double* A, int n1, double* B, int n2, double* C)
{
	int merged_array_size = n1 + n2;
	int a = 0;
	int b = 0;
	int i = 0;

	for (; i < merged_array_size; i++)
	{
		if (A[a] < B[b])
		{
			C[i] = A[a];
			a++;
		}
		else
		{
			C[i] = B[b];
			b++;
		}

		if (a == n1 || b == n2)
		{
			break;
		}
	}

	if (a < n1)
	{
		for (i++; i < merged_array_size; i++)
		{
			C[i] = A[a];
			a++;
		}
	}

	if (b < n2)
	{
		for (i++; i < merged_array_size; i++)
		{
			C[i] = B[b];
			b++;
		}
	}

	delete[] A;
	A = NULL;
	
	delete[] B;
	B = NULL;
}

/*Mergesort.*/
void sort(double* S, int n)
{
	if (n != 1)
	{
		int mid = n / 2;
		double* S1 = new double[mid];
		double* S2 = new double[n - mid];

		for (int i = 0; i < mid; i++)
		{
			S1[i] = S[i];
		}

		for (int i = 0; i < n - mid; i++)
		{
			S2[i] = S[i + mid];
		}

		sort(S1, mid);
		sort(S2, n - mid);
		
		merge(S1, mid, S2, n - mid, S);
	}
}

/*Gets a subarray of a given size.*/
void get_subarray(double* S, int n, double* subarray, int subarray_number, int elements)
{
	int starting_point = subarray_number * elements;
	for (int i = 0; i < elements && starting_point + i < n; i++)
	{
		subarray[i] = S[starting_point + i];
	}
}

/*Compares two values.*/
bool compare(double a, double b, int comparison)
{
	switch (comparison)
	{
	case LESS:
		return a < b;
	case EQUALS:
		return a == b;
	case GREATER:
		return a > b;
	default:
		return false;
	}
}

/*Returns the amount of elements less, equal, or greater than m 
based on the given comparison.*/
int get_split_array_size(double* S, int n, double m, int comparison)
{
	int size = 0;
	for (int i = 0; i < n; i++)
	{
		if (compare(S[i], m, comparison))
		{
			size++;
		}
	}

	return size;
}

/*Creates a subarray with elements less, equal, or greater than m 
based on the given comparison.*/
double* split_array(double* S, int n, double m, int comparison)
{
	int size = get_split_array_size(S, n, m, comparison);

	double* S_new = new double[size];
	int j = 0;
	for (int i = 0; i < n; i++)
	{
		if (compare(S[i], m, comparison))
		{
			S_new[j] = S[i];
			j++;
		}
	}

	return S_new;
}

/*Sequential Select. Returns the kth element of an array S of size n.*/
double sequential_select(double* S, int n, int k)
{
	if (n >= Q)
	{
		sort(S, n);
		double selected = S[k];
		
		delete[] S;
		S = NULL;

		return selected;
	}
	else
	{
		int subarrays_count = n % Q == 0 ? n / Q : n / Q + 1;
		double* M = new double[subarrays_count];
		for (int i = 0; i < subarrays_count; i++)
		{
			double* subarray = new double[Q];
			get_subarray(S, n, subarray, i, Q);
			M[i] = sequential_select(subarray, Q, Q / 2);

			delete[] subarray;
			subarray = NULL;
		}

		double m = sequential_select(M, subarrays_count, subarrays_count / 2);
		
		delete[] M;
		M = NULL;

		int S1_size = get_split_array_size(S, n, m, LESS);
		int S2_size = get_split_array_size(S, n, m, EQUALS);
		int S3_size = get_split_array_size(S, n, m, GREATER);

		if (S1_size >= k)
		{
			double* S1 = split_array(S, n, m, LESS);
			return sequential_select(S1, S1_size, k);
		}
		else if (S1_size + S2_size >= k)
		{
			return m;
		}
		else
		{
			double* S3 = split_array(S, n, m, GREATER);
			return sequential_select(S3, S3_size, k - S1_size - S2_size);
		}
	}
}

/*Parallel Select. Not fully implemented.*/
double parallel_select(double* S, int n, int k)
{
	int rank = omp_get_thread_num();
	int threads = omp_get_num_threads();
	if (n <= Q)
	{
		if (rank == 0)
		{
			sort(S, n);
			double selected = S[k];

			delete[] S;
			S = NULL;

			return selected;
		}
	}
	else 
	{
		double x = 1 - std::log(threads)/std::log(n);
		int subarray_size = std::pow(n, x);
		int subarrays_count = n / subarray_size;
		double* M = new double[subarrays_count];
		
		double* subarray = new double[subarray_size];
		get_subarray(S, n, subarray, rank, subarray_size);
		M[rank] = sequential_select(subarray, subarray_size, subarray_size / 2);

		double m = parallel_select(M, subarrays_count, subarrays_count / 2);

		int L_size, E_size, G_size;
#pragma omp barrier
		if (rank == 0)
		{
			L_size = 0, E_size = 0, G_size = 0;
		}

#pragma omp critical
		L_size += get_split_array_size(subarray, subarray_size, m, LESS);

#pragma omp critical
		E_size += get_split_array_size(subarray, subarray_size, m, EQUALS);

#pragma omp critical
		G_size += get_split_array_size(subarray, subarray_size, m, GREATER);

		delete[] subarray;
		subarray = NULL;

		delete[] M;
		M = NULL;

		if (L_size >= k)
		{
			double* L = split_array(S, n, m, LESS);
			return parallel_select(L, L_size, k);
		}
		else if (L_size + E_size >= k)
		{
			return m;
		}
		else
		{
			double* G = split_array(S, n, m, GREATER);
			return parallel_select(G, G_size, k - L_size - E_size);
		}
	}
}

/*Prints the kth element of the array, the method used, 
the threads used and the time it required.*/
void print_results(double result, int k, int method, std::string method_name, int threads, double time)
{
	std::string ordinal;
	switch (k%10)
	{
	case 1:
		ordinal = "st";
		break;
	case 2:
		ordinal = "nd";
		break;
	case 3:
		ordinal = "rd";
	default:
		ordinal = "th";
		break;
	}

	std::cout << "The " << k << ordinal << " element of the array is " << result << "." << std::endl;
	std::cout << "Time required for " << method_name << " was " << time << "." << std::endl;

	if (method == 1)
	{
		std::cout << "Threads used: " << threads;
	}
}

/*Picks default values for the program.*/
void set_default_input(int* input)
{
	input[0] = DEFAULT_SIZE;
	input[1] = DEFAULT_SELECTED;
	input[2] = 0;
	input[3] = 1;
}

/*Verifies the validity of given parameteres.
If they are wrong, it calls set_default_input.*/
void manage_input(int argc, char* argv[], int* input)
{
	//Not enough parameters
	if (argc < 4)
	{
		std::cout << "Not enough parameters. " << std::endl;
		std::cout << "Arguments are n, k, method and threads. n is the " <<
			"amount of elements of the array, " <<
			"k is the kth smallest element of the array " <<
			"and method is 0(Sequential Select) " <<
			"or 1(Parallel Select). threads is used for parallel select. " << std::endl << std::endl;

		std::cout << "Setting size to " << DEFAULT_SIZE << " and selected element to " << DEFAULT_SELECTED <<
			". Selecting Sequential Select." << std::endl << std::endl;
		set_default_input(input);
		return;
	}

	input[0] = atoi(argv[1]);
	input[1] = atoi(argv[2]);
	input[2] = atoi(argv[3]);

	//n<k or n<0 or k<0
	bool invalid_n_k = input[0] < input[1] || input[0] <= 0 || input[1] <= 0;
	if (invalid_n_k)
	{
		std::cout << "Invalid combination of n and k. Both need to be positive with n >= k. " <<
			"Setting size to " << DEFAULT_SIZE << " and selected element to " << DEFAULT_SELECTED <<
			". Selecting Sequential Select." << std::endl << std::endl;
		set_default_input(input);
		return;
	}

	//Method not selected properly
	bool invalid_method = input[2] != 0 && input[2] != 1;
	if (invalid_method)
	{
		std::cout << "Invalid method. 0 for Sequential Select and 1 for Parallel Select. " <<
			"Setting size to " << DEFAULT_SIZE << " and selected element to " << DEFAULT_SELECTED <<
			". Selecting Sequential Select." << std::endl << std::endl;
		set_default_input(input);
		return;
	}

	//Invalid threads
	if (argc >= 5)
	{
		int threads = atoi(argv[4]);
		input[3] = threads > 0 ? threads : 1;
	}
	else
	{
		input[3] = 1;
	}
		
}

int main(int argc, char* argv[])
{
	int* input = new int[4]; 
	manage_input(argc, argv, input);

	int n = input[0];
	int k = input[1];
	int method = input[2];
	int thread_count = input[3];

	std::string method_name;

	switch (method)
	{
	case 0:
		method_name = "Sequential Select";
		break;
	case 1:
		method_name = "Parallel Select";
		break;
	default:
		break;
	}

#pragma omp barrier
	double start = omp_get_wtime();

	double* A = new double[n];
	create_array(A, n);	

	//Used on small arrays to verify result
	//print_array(A, n); 

	double result;
	switch (method)
	{
	case 0:
		result = sequential_select(A, n, k);
		break;
	case 1:
#pragma omp parallel num_threads(thread_count)
		result = parallel_select(A, n, k);
		break;
	default:
		break;
	}

#pragma omp barrier
	double end = omp_get_wtime();

	print_results(result, k, method, method_name, thread_count, end - start);

	delete[] input;
	input = NULL;

	return 0;
}