
using namespace std;

//-------

#include <piston/halo_naive.h>
#include <piston/halo_kd.h>
#include <piston/halo_vtk.h>
#include <piston/halo_merge.h>

//-------

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

using namespace piston;

/*
struct compare
{
	int *a, *b, *c;

	__host__ __device__
	compare(int* a, int* b, int* c) :
		a(a), b(b), c(c) {}

	__host__ __device__
	void operator()(int i)
	{
		if(a[i] != b[i])
			c[i] = 1;
	}
};

bool compareResults(thrust::device_vector<int> a, thrust::device_vector<int> b, int numOfParticles)
{
	thrust::device_vector<int> c(numOfParticles);
	thrust::fill(c.begin(), c.end(), 0);

	thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
			compare(thrust::raw_pointer_cast(&*a.begin()), thrust::raw_pointer_cast(&*b.begin()), thrust::raw_pointer_cast(&*c.begin())));
	int result = thrust::reduce(c.begin(), c.begin() + numOfParticles);

	if(result==0)
		return true;

std::cout << result << std::endl;

	return false;
}

int main(int argc, char* argv[])
{

  halo *halo;

  float linkLength, max_linkLength, min_linkLength;
  int   particleSize, rL, np, n;

  max_linkLength = 2;
  min_linkLength = 0;
  linkLength     = 0.2;
  particleSize   = 100;
  np = 256;
  rL = 64;
  n  = 1; //if you want a fraction of the file to load, use this.. 1/n

  char filename[1024];
// sprintf(filename, "%s/sub-8435", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
// std::string format = "csv";
  sprintf(filename, "%s/sub-24474", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
  std::string format = "csv";
//  sprintf(filename, "%s/256", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
//  std::string format = "cosmo";

  std::cout << "min_linkLength " << min_linkLength << std::endl;
  std::cout << "max_linkLength " << max_linkLength << std::endl;
  std::cout << "linkLength " << linkLength << std::endl;
  std::cout << "particleSize " << particleSize << std::endl;
  std::cout << filename << std::endl;
  std::cout << std::endl;

  //----------------------------

//  std::cout << "Naive result" << std::endl;
//
//  halo = new halo_naive(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> a = halo->getHalos();
//
//  std::cout << "VTK based result" << std::endl;
//
//  halo = new halo_vtk(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> b = halo->getHalos();
//
//  std::cout << "Kdtree based result" << std::endl;
//
//  halo = new halo_kd(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> c = halo->getHalos();

  std::cout << "Merge tree based result" << std::endl;

  halo = new halo_merge(min_linkLength, max_linkLength, filename, format, n, np, rL);
  (*halo)(linkLength, particleSize);
  thrust::device_vector<int> d = halo->getHalos();

  //----------------------------

//  std::cout << "Comparing results" << std::endl;
//  std::string output1 = (compareResults(a, c, halo->numOfParticles)==true) ? "Naive vs Kdtree     - Result is the same" : "Naive vs Kdtree        - Result is NOT the same";
//  std::cout << output1 << std::endl;
//  std::string output2 = (compareResults(b, c, halo->numOfParticles)==true) ? "Vtk vs Kdtree     - Result is the same" : "Vtk vs Kdtree     - Result is NOT the same";
//  std::cout << output2 << std::endl;
//  std::string output3 = (compareResults(c, d, halo->numOfParticles)==true) ? "Kdtree vs Mergetree - Result is the same" : "Kdtree vs Mergetree - Result is NOT the same";
//  std::cout << output3 << std::endl;
//  std::cout << "--------------------" << std::endl;

//	std::cout << "a "; thrust::copy(a.begin(), a.begin()+163, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
//  std::cout << "c "; thrust::copy(c.begin(), c.begin()+163, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
//  std::cout << "d "; thrust::copy(d.begin(), d.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;

  return 0;
}
*/

//------------------------

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <iostream>

// This example shows how to perform a lexicographical sort on multiple keys.
//
// http://en.wikipedia.org/wiki/Lexicographical_order

template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation)
{
  // temporary storage for keys
  KeyVector temp(keys.size());

  // permute the keys with the current reordering
  thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

  struct timeval begin, end, diff;

  // stable_sort the permuted keys and update the permutation
  gettimeofday(&begin, 0);
  thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
  gettimeofday(&end, 0);

  timersub(&end, &begin, &diff);
  float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
  std::cout << "Points: " << keys.size() << " SORT Time: " << seconds << std::endl << std::flush;
}

template <typename KeyVector, typename PermutationVector>
void apply_permutation(KeyVector& keys, PermutationVector& permutation)
{
  // copy keys to temporary vector
  KeyVector temp(keys.begin(), keys.end());

  // permute the keys
  thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}

thrust::host_vector<int> random_vector(size_t N)
{
  thrust::host_vector<int> vec(N);
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9);

  for (size_t i = 0; i < N; i++)
  vec[i] = dist(rng);

  return vec;
}

thrust::host_vector<float> random_vectorF(size_t N)
{
  thrust::host_vector<float> vec(N);
  static thrust::default_random_engine rng;
  static thrust::uniform_real_distribution<float> dist(0, 9);

  for (size_t i = 0; i < N; i++)
  vec[i] = dist(rng);

  return vec;
}

int main(void)
{
  for(int i=0; i<20; i++)
  {
    size_t N = 100000*i;

    // generate three arrays of random values
    thrust::device_vector<int> upper = random_vector(N);

    //std::cout << "Unsorted Keys" << std::endl;
    //for(size_t i = 0; i < N; i++)
    //{
    //std::cout << "(" << upper[i] << "," << middle[i] << "," << lower[i] << ")" << std::endl;
    //}

    // initialize permutation to [0, 1, 2, ... ,N-1]
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    // sort from least significant key to most significant keys
    update_permutation(upper, permutation);

    // Note: keys have not been modified
    // Note: permutation now maps unsorted keys to sorted order

    // permute the key arrays by the final permuation
    apply_permutation(upper, permutation);

    //std::cout << "Sorted Keys" << std::endl;
    //for(size_t i = 0; i < N; i++)
    //{
    //std::cout << "(" << upper[i] << "," << middle[i] << "," << lower[i] << ")" << std::endl;
    //}
  }

  return 0;
}
