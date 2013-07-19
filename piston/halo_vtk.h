#ifndef HALO_VTK_H_
#define HALO_VTK_H_

#include <piston/halo.h>

#define numDataDims 3
#define dataX 0
#define dataY 1
#define dataZ 2

namespace piston
{

class halo_vtk : public halo
{
public:		
	typedef thrust::tuple<float, float> BoundsTuple;
	
	typedef thrust::tuple<int, float>  		   				KdtreeTuple;
	typedef thrust::tuple<IntIterator, FloatIterator>       KdtreeTupleIterator;
	typedef thrust::zip_iterator<KdtreeTupleIterator>       KdtreeTupleZipIterator;
		
	thrust::device_vector<BoundsTuple>  boundsX, boundsY, boundsZ;
	KdtreeTupleZipIterator 				firstKD, lastKD;
	
    thrust::device_vector<int>  halos;
    thrust::device_vector<int> nextp;

	halo_vtk(std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1): halo(filename, format, n, np, rL) {}
	
	void operator()(float linkLength , int  particleSize)		
	{
		clear();

		linkLength    = linkLength;
		particleSize  = particleSize;

		// no valid particles, return
		if(numOfParticles==0) return;

		// set the vectors for creating balanced-kdtre
		thrust::device_vector<int> ind(numOfParticles);
		thrust::sequence(ind.begin(), ind.end());
		thrust::device_vector<float> val(numOfParticles);
		thrust::fill(val.begin(), val.end(), 0);

		// get the first & last of the zip iterator
		firstKD = thrust::make_zip_iterator(thrust::make_tuple(ind.begin(), val.begin()));
		lastKD  = thrust::make_zip_iterator(thrust::make_tuple(ind.end(),   val.end()));

		struct timeval begin, end, diff;
		gettimeofday(&begin, 0);

		// create the balanced-kdtree
		getBalancedKdTree(0, numOfParticles, dataX, firstKD);

		gettimeofday(&end, 0);

		timersub(&end, &begin, &diff);
		float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		std::cout << "Time elapsed: " << seconds << " s for KD tree construction"<< std::endl;

		// set the size for computing bounds
		boundsX = thrust::device_vector<BoundsTuple>(numOfParticles);
		boundsY = thrust::device_vector<BoundsTuple>(numOfParticles);
		boundsZ = thrust::device_vector<BoundsTuple>(numOfParticles);

		gettimeofday(&begin, 0);

		// compute bound of each particle
		computeBounds(0, numOfParticles, firstKD);

		gettimeofday(&end, 0);

		timersub(&end, &begin, &diff);
		seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		std::cout << "Time elapsed: " << seconds << " s for computing bounds"<< std::endl;

		// set vectors for halo finding
		halos  = thrust::device_vector<int>(numOfParticles);
		thrust::sequence(halos.begin(), halos.end());
		nextp = thrust::device_vector<int>(numOfParticles);
		thrust::fill(nextp.begin(), nextp.end(), -1);

		gettimeofday(&begin, 0);

		// find halos
		findHalos(0, numOfParticles, dataX, firstKD, linkLength, numOfParticles, false/*periodic*/);

		gettimeofday(&end, 0);

		timersub(&end, &begin, &diff);
		seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		std::cout << "Time elapsed: " << seconds << " s for merging"<< std::endl;

		gettimeofday(&begin, 0);

		getUniqueHalos(particleSize); // get the unique valid halo ids

		gettimeofday(&end, 0);

		timersub(&end, &begin, &diff);
		seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		std::cout << "Time elapsed: " << seconds << " s for finding valid halos"<< std::endl;

		setColors(); // set colors to halos

		std::cout << "Number of Particles : " << numOfParticles <<  " Number of Halos found : " << numOfHalos << std::endl << std::endl;
	}

protected:

	//----------- methods for creating B-kd tree
	void getBalancedKdTree(int start, int end, int dim, KdtreeTupleZipIterator firstKD)
	{
		int size = end - start;

		if(size ==1) return;

		thrust::device_vector<float> data;
		if(dim == 0) data = inputX;
		else if(dim == 1) data = inputY;
		else if(dim == 2) data = inputZ;
		
		int half = (float)(size/2);

		thrust::device_vector<float> keys(size);
		thrust::device_vector<int>   vals(size);
		for(int i=0; i<size; i++)
		{
			vals[i] = thrust::get<0>(firstKD[i+start]);
			keys[i] = data[thrust::get<0>(firstKD[i+start])];
		}

		thrust::sort_by_key(keys.begin(), keys.begin()+size, vals.begin()); // this sorts the whole array, but we just need the nth element in the sorted list

		for(int i=0; i<size; i++) firstKD[i+start] = thrust::make_tuple(vals[i], keys[i]);

		// divide 
		getBalancedKdTree(start,      start+half, (dim+1)%3, firstKD);
		getBalancedKdTree(start+half, end,        (dim+1)%3, firstKD);
	}

	//----------- methods for computing bounds
	void computeBounds(int start, int end, KdtreeTupleZipIterator firstKD)
	{
		int size  = end - start;

		int middle  = start + (float)(size/2);
		int middle1 = start + (float)(size/4);
		int middle2 = start + (float)(3*size/4);

		if(size==2)
		{
			int ii = thrust::get<0>(firstKD[start]);
			int jj = thrust::get<0>(firstKD[start+1]);
			
			boundsX[middle] = thrust::make_tuple(std::min(inputX[ii], inputX[jj]), std::max(inputX[ii], inputX[jj]));
			boundsY[middle] = thrust::make_tuple(std::min(inputY[ii], inputY[jj]), std::max(inputY[ii], inputY[jj]));
			boundsZ[middle] = thrust::make_tuple(std::min(inputZ[ii], inputZ[jj]), std::max(inputZ[ii], inputZ[jj]));
			
			return;
		}

		if(size==3)
		{
			computeBounds(start+1, end, firstKD);

			int ii = thrust::get<0>(firstKD[start]);
			
			boundsX[middle] = thrust::make_tuple(std::min((float)inputX[ii], thrust::get<0>((BoundsTuple)boundsX[middle2])), std::max((float)inputX[ii], thrust::get<1>((BoundsTuple)boundsX[middle2])));
			boundsY[middle] = thrust::make_tuple(std::min((float)inputY[ii], thrust::get<0>((BoundsTuple)boundsY[middle2])), std::max((float)inputY[ii], thrust::get<1>((BoundsTuple)boundsY[middle2])));
			boundsZ[middle] = thrust::make_tuple(std::min((float)inputZ[ii], thrust::get<0>((BoundsTuple)boundsZ[middle2])), std::max((float)inputZ[ii], thrust::get<1>((BoundsTuple)boundsZ[middle2])));
									
			return;
		}

		computeBounds(start,  middle, firstKD);
		computeBounds(middle, end,    firstKD);

		boundsX[middle] = thrust::make_tuple(std::min(thrust::get<0>((BoundsTuple)boundsX[middle1]), thrust::get<0>((BoundsTuple)boundsX[middle2])), std::max(thrust::get<1>((BoundsTuple)boundsX[middle1]), thrust::get<1>((BoundsTuple)boundsX[middle2])));
		boundsY[middle] = thrust::make_tuple(std::min(thrust::get<0>((BoundsTuple)boundsY[middle1]), thrust::get<0>((BoundsTuple)boundsY[middle2])), std::max(thrust::get<1>((BoundsTuple)boundsY[middle1]), thrust::get<1>((BoundsTuple)boundsY[middle2])));
		boundsZ[middle] = thrust::make_tuple(std::min(thrust::get<0>((BoundsTuple)boundsZ[middle1]), thrust::get<0>((BoundsTuple)boundsZ[middle2])), std::max(thrust::get<1>((BoundsTuple)boundsZ[middle1]), thrust::get<1>((BoundsTuple)boundsZ[middle2])));

		return;
	}
	
	//----------- methods for merging halos
	void findHalos(int start, int end, int dim, KdtreeTupleZipIterator firstKD, float linkLength, int numOfParticles, bool periodic)
	{
		int size = end - start;

		if(size==1) return;

		int middle = start + (float)(size/2);

		findHalos(start,  middle, (dim+1)%3, firstKD, linkLength, numOfParticles, periodic);
		findHalos(middle, end,    (dim+1)%3, firstKD, linkLength, numOfParticles, periodic);

		mergeHalos(start, middle, middle, end, dim, firstKD, linkLength, numOfParticles, periodic);

		return;
	}
	
	void mergeHalos(int start1, int end1, int start2, int end2, int dim, KdtreeTupleZipIterator firstKD, float linkLength, int numOfParticles, bool periodic)
	{
		int size1 = end1 - start1;
		int size2 = end2 - start2;
		
		if(size1==1 || size2==1)
		{
			for(int i=0; i<size1; i++)
			{
				for(int j=0; j<size2; j++)
				{
					int ii = thrust::get<0>(firstKD[start1+i]);
					int jj = thrust::get<0>(firstKD[start2+j]);
										
					if(haloIndex[ii] == haloIndex[jj]) continue;
					
					float xd = std::fabs(inputX[jj] - inputX[ii]);
					float yd = std::fabs(inputY[jj] - inputY[ii]);
					float zd = std::fabs(inputZ[jj] - inputZ[ii]);

					if (periodic)
					{
						xd = std::min(xd, numOfParticles-xd);
						yd = std::min(yd, numOfParticles-yd);
						zd = std::min(zd, numOfParticles-zd);
					}
					
					if(xd<=linkLength && yd<=linkLength && zd<=linkLength)
					{
						float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);
						if(dist <= linkLength)
						{
							int newHaloId = std::min(haloIndex[ii], haloIndex[jj]);
							int oldHaloId = std::max(haloIndex[ii], haloIndex[jj]);
														
							int last = -1;
							int ith = halos[oldHaloId];
							while(ith != -1)
							{
								haloIndex[ith] = newHaloId;
								last = ith;
								ith  = nextp[ith];
							}
							
							nextp[last]      = halos[newHaloId];
							halos[newHaloId] = halos[oldHaloId];
							halos[oldHaloId] = -1;
						}	
					}
				}
			}
			return;
		}
		
		int middle1 = start1 + (float)(size1/2);
		int middle2 = start2 + (float)(size2/2);
		
		thrust::device_vector<BoundsTuple> bounds;
		if(dim == 0) bounds = boundsX;
		else if(dim == 1) bounds = boundsY;
		else if(dim == 2) bounds = boundsZ;
		
		float lL = thrust::get<0>((BoundsTuple)bounds[middle1]);
		float uL = thrust::get<1>((BoundsTuple)bounds[middle1]);
		float lR = thrust::get<0>((BoundsTuple)bounds[middle2]);
		float uR = thrust::get<1>((BoundsTuple)bounds[middle2]);

		float dL = uL - lL;
		float dR = uR - lR;
		float dc = std::max(uL,uR) - std::min(lL,lR);
		
		float dist = dc - dL - dR;
		if(periodic) dist = std::min(dist, numOfParticles-dc);
		
		if(dist>linkLength) return;
		
		dim = (dim+1)%3;
		
		mergeHalos(start1,  middle1, start2,  middle2, dim, firstKD, linkLength, numOfParticles, periodic);
		mergeHalos(start1,  middle1, middle2, end2,    dim, firstKD, linkLength, numOfParticles, periodic);
		mergeHalos(middle1, end1,    start2,  middle2, dim, firstKD, linkLength, numOfParticles, periodic);
		mergeHalos(middle1, end1,    middle2, end2,    dim, firstKD, linkLength, numOfParticles, periodic);
		
		return;
	}
};

}

#endif
