#ifndef HALO_NAIVE_H_
#define HALO_NAIVE_H_

#include <piston/halo.h>

#include <cmath>

namespace piston
{

class halo_naive : public halo
{
public:

	halo_naive(std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1) : halo(filename, format, n, np, rL)
		{}

	void operator()(float linkLength , int  particleSize)
	{
		clear();

		linkLength    = linkLength;
		particleSize  = particleSize;

		// no valid particles, return
		if(numOfParticles==0)
			return;

		// get the first & last of the zip iterator
		ParticleTupleZipIterator first = thrust::make_zip_iterator(thrust::make_tuple(inputX.begin(), inputY.begin(), inputZ.begin()));
		ParticleTupleZipIterator last  = thrust::make_zip_iterator(thrust::make_tuple(inputX.end(),   inputY.end(),   inputZ.end()));

		struct timeval begin, mid, end, diff1, diff2;
		gettimeofday(&begin, 0);

		// for each particle, get find particles in it's halo
		thrust::device_vector<int> result   (numOfParticles);
		thrust::device_vector<int> resultTmp(numOfParticles);
		for(unsigned int i=0; i<numOfParticles; i++)
		{
			// get all the valid particles for the current particle
			thrust::transform(CountingIterator(0), CountingIterator(0)+numOfParticles, result.begin(),
					inTheSameHalo(thrust::raw_pointer_cast(&*inputX.begin()), thrust::raw_pointer_cast(&*inputY.begin()), thrust::raw_pointer_cast(&*inputZ.begin()),
							 i, linkLength));
			IntIterator new_end = thrust::remove(result.begin(), result.end(), -1);

			// go through all such particles & set halo ids, merge halos if needed
			for(unsigned int j=0; j<(new_end - result.begin()); j++)
			{
				int current = result[j];

				ParticleTuple a = first[i];
				ParticleTuple b = first[current];

				// merge halos
				if(haloIndex[i] != haloIndex[current])
				{
					int min = std::min(haloIndex[i], haloIndex[current]);

					int id = (min==haloIndex[i]) ? current : i;
					thrust::transform(CountingIterator(0), CountingIterator(0)+numOfParticles, resultTmp.begin(), check_equal(id, thrust::raw_pointer_cast(&*haloIndex.begin())));
					IntIterator new_end = thrust::remove(resultTmp.begin(), resultTmp.end(), -1);

					thrust::for_each(resultTmp.begin(), new_end, set_haloId(min, thrust::raw_pointer_cast(&*haloIndex.begin())));
				}
			}
		}

		gettimeofday(&mid, 0);

		getUniqueHalos(particleSize); // get the unique valid halo ids

		gettimeofday(&end, 0);

		timersub(&mid, &begin, &diff1);
		float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
		std::cout << "Time elapsed: " << seconds1 << " s for finding halos"<< std::endl;
		timersub(&end, &mid, &diff2);
		float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
		std::cout << "Time elapsed: " << seconds2 << " s for finding valid halos"<< std::endl;

		setColors(); // set colors to halos

		std::cout << "Number of Particles : " << numOfParticles <<  " Number of Halos found : " << numOfHalos << std::endl << std::endl;
	}

	// given two particles, find whether they are in the same halo
	struct inTheSameHalo : public thrust::unary_function<int, int>
	{
		float linkLength;
		int i;
		float *inputX, *inputY, *inputZ;

		__host__ __device__
		inTheSameHalo(float *inputX, float *inputY, float *inputZ, int i, float linkLength) :
			inputX(inputX), inputY(inputY), inputZ(inputZ), i(i), linkLength(linkLength)
		    {}

		__host__ __device__
		int operator()(int j)
		{
			float xd   = std::fabs(inputX[j] - inputX[i]);
			float yd   = std::fabs(inputY[j] - inputY[i]);
			float zd   = std::fabs(inputZ[j] - inputZ[i]);
			float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);

			if(xd<=linkLength && yd<=linkLength && zd<=linkLength)
				if (dist<=linkLength)
					return j;

			return -1;
		}
	};

	// given two particles, check whether their halo ids are same
	struct check_equal : public thrust::unary_function<int, int>
	{
		int i;
		int *haloIndex;

		__host__ __device__
		check_equal(int i, int *haloIndex) : i(i), haloIndex(haloIndex) {}

		__host__ __device__
		int operator()(int j)
		{
			if (haloIndex[i] == haloIndex[j])
				return j;

			return -1;
		}
	};

	// set halo id of particle a to particle b
	struct set_haloId : public thrust::unary_function<int, void>
	{
		int i;
		int *haloIndex;

		__host__ __device__
		set_haloId(int i, int *haloIndex) : i(i), haloIndex(haloIndex) {}

		__host__ __device__
		void operator()(int j)
		{
			haloIndex[j] = haloIndex[i];
		}
	};
};

}

#endif
