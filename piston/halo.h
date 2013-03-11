#ifndef HALO_H_
#define HALO_H_

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/merge.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <piston/kd.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>
#include <cmath>


namespace piston
{

class halo
{
public:

	struct DataItem
	{
		float xx;  float yy;  float zz;
	    float vx;  float vy;  float vz;
	    float ms;  int   pt;

	    __host__ __device__
	    DataItem(){}

	    __host__ __device__
	    DataItem(float xx, float vx, float yy, float vy, float zz, float vz, float ms, int pt) :
		     xx(xx), vx(vx), yy(yy), vy(vy), zz(zz), vz(vz), ms(ms), pt(pt) {}
	};

	struct Point
	{
	    float x, y, z;

	    __host__ __device__
	    Point(){}

	    __host__ __device__
	    Point(float x, float y, float z) : x(x), y(y), z(z) {}
	};

    // definitions
    typedef thrust::device_vector<float>::iterator FloatIterator;
    typedef thrust::device_vector<int>::iterator IntIterator;

    typedef thrust::tuple<float, float> Float2;

    typedef thrust::tuple<float, float, float> Float3;
    typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> Float3TupleIterator;
    typedef thrust::zip_iterator<Float3TupleIterator> Float3zipIterator;

    typedef Float3 ParticleTuple;
    typedef Float3TupleIterator ParticleTupleIterator;
    typedef Float3zipIterator ParticleTupleZipIterator;

    typedef thrust::counting_iterator<int> CountingIterator;

    //inputs
    float linkLength;     // linking length used to link two particles
    int   particleSize;   // number of particles in a halo
    int   numOfParticles; // total number of particles
    float rL;
    int   np;             // number of particles in a dimension
    bool  periodic;
    int   numOfHalos;     // total number of halos

    thrust::device_vector<float>  inputX, inputY, inputZ;		 // positions for each particle
    thrust::device_vector<int>    haloIndex;	  			     // halo indices for each particle
    thrust::device_vector<int>    haloIndexUnique;          	 // unique halo indexes
    thrust::device_vector<float>  haloColorsR, haloColorsG, haloColorsB; // colors for each halo

    // variables needed to create random numbers
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01;

    halo(std::string filename="", std::string format=".cosmo", int n=1, int np=1, float rL=-1, bool periodic=false)
    {
        u01 = thrust::uniform_real_distribution<float>(0.0f, 1.0f);

        rL       = rL;
        np       = np;
        periodic = periodic;

        if(!readFile(filename, rL, np, n, format))
        {
			numOfParticles = 8;
			inputX = thrust::host_vector<float>(numOfParticles);
			inputY = thrust::host_vector<float>(numOfParticles);
			inputZ = thrust::host_vector<float>(numOfParticles);

			inputX[0] = 1.0;	inputY[0] = 1.0;	inputZ[0] = 0.0;
			inputX[1] = 0.0;	inputY[1] = 4.0;	inputZ[1] = 0.0;
			inputX[2] = 8.0;	inputY[2] = 3.0;	inputZ[2] = 0.0;
			inputX[3] = 2.0;	inputY[3] = 6.0;	inputZ[3] = 0.0;
			inputX[4] = 4.0;	inputY[4] = 2.0;	inputZ[4] = 0.0;
			inputX[5] = 5.0;	inputY[5] = 5.0;	inputZ[5] = 0.0;
			inputX[6] = 3.0;	inputY[6] = 4.0;	inputZ[6] = 0.0;
			inputX[7] = 3.5;	inputY[7] = 4.5;	inputZ[7] = 0.0;

			std::cout << "Test data loaded \n";
        }

        std::cout << "numOfParticles : " << numOfParticles << " \n";
    }


    virtual void operator()(float linkLength , int  particleSize) {}


    // get start of vertices
    Float3zipIterator vertices_begin()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(inputX.begin(), inputY.begin(), inputZ.begin()));
    }


    // get end of vertices
    Float3zipIterator vertices_end()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(inputX.end(), inputY.end(), inputZ.end()));
    }


    // get the color of halo i
    Float3 getColor(int i)
    {
        return thrust::make_tuple(haloColorsR[i], haloColorsG[i], haloColorsB[i]);
    }


    // get the index in haloIndexUnique for a halo i
    int getHaloInd(int i)
    {
        IntIterator ite = thrust::find(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfHalos, haloIndex[i]);
        return (ite!=haloIndexUnique.begin()+numOfHalos) ? ite - haloIndexUnique.begin() : -1;
    }


    // read input file - currently can read a .cosmo file or a .csv file
    // .csv file usage - when you have a big data file and you want a piece of it, load it in VTK and slice it and save it as .csv file
    bool readFile(std::string filename, float rL, int np, float n, std::string format)
    {
        // check filename
        if(filename == "") { std::cout << "no input file specified \n"; return false; }

        if(format=="csv")    return readCsvFile(filename, rL, np, n, format);

        if(format=="cosmo")  return readCosmoFile(filename, rL, np, n, format);

        return false;
    }

    // read a .cosmo file and load the data to inputX, inputY & inputZ
	bool readCosmoFile(std::string filename, float rL, int np, float n, std::string format)
	{
		std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
		if (!myfile->is_open()) {   std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

		// compute the number of particles
		myfile->seekg(0L, std::ios::end);
		numOfParticles = myfile->tellg() / 32;  // get particle size in file
		numOfParticles = numOfParticles / n;    // get the fraction wanted

		 // resize inputX, inputY, inputZ temp vectors
		thrust::host_vector<float> inputXHost(numOfParticles);
		thrust::host_vector<float> inputYHost(numOfParticles);
		thrust::host_vector<float> inputZHost(numOfParticles);

		// scale amount for particles
		float xscal;
		if(rL==-1) xscal = 1;
		else       xscal = rL / (1.0*np);

		// rewind file to beginning for particle reads
		myfile->seekg(0L, std::ios::beg);

		// declare temporary read buffers
		int nfloat = 7, nint = 1;
		float fBlock[nfloat];
		int iBlock[nint];

		for (int i=0; i<numOfParticles; i++)
		{
			// Set file pointer to the requested particle
			myfile->read((char *)fBlock, nfloat * sizeof(float));

			if (myfile->gcount() != (int)(nfloat * sizeof(float))) {
			  std::cout << "Premature end-of-file" << std::endl;
			  return false;
			}

			myfile->read((char *)iBlock, nint * sizeof(int));
			if (myfile->gcount() != (int)(nint * sizeof(int))) {
			  std::cout << "Premature end-of-file" << std::endl;
			  return false;
			}

			// These files are always little-endian
			//vtkByteSwap::Swap4LERange(fBlock, nfloat);
			//vtkByteSwap::Swap4LERange(iBlock, nint);

			// sanity check
			if (fBlock[0] > rL || fBlock[2] > rL || fBlock[4] > rL) {
			  std::cout << "rL is too small" << std::endl;
			  exit (-1);
			}

			inputXHost[i] = fBlock[0] / xscal;
			inputYHost[i] = fBlock[2] / xscal;
			inputZHost[i] = fBlock[4] / xscal;

			//std::cout << inputXHost[i] << " " << inputYHost[i] << " " << inputZHost[i] << "\n";
		}

		inputX = inputXHost;
		inputY = inputYHost;
		inputZ = inputZHost;

		return true;
	}


    // read a .csv file and load the data to inputX, inputY & inputZ
    bool readCsvFile(std::string filename, float rL, int np, float n, std::string format)
    {
        // open input file
        std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
        if (!myfile->is_open()) {   std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

        std::vector<Point> vec;
        float x, y, z;

        std::string line;
        getline(*myfile,line);

        while(!myfile->eof())
        {
			getline(*myfile,line);

			if(line=="") continue;
			strtok((char*)line.c_str(),",");

			int i = 0;
			while(++i<7) strtok (NULL, ",");

			x = atof(strtok(NULL, ","));
			y = atof(strtok(NULL, ","));
			z = atof(strtok(NULL, ","));

			vec.push_back(Point(x,y,z));
        }

        numOfParticles = vec.size() / n;

        // resize inputX, inputY, inputZ temp vectors
        thrust::host_vector<float> inputXHost(numOfParticles);
        thrust::host_vector<float> inputYHost(numOfParticles);
        thrust::host_vector<float> inputZHost(numOfParticles);

        // scale amount for particles
        float xscal;
        if(rL==-1) xscal = 1;
        else       xscal = rL / (1.0*np);

        for (int i=0; i<numOfParticles; i++)
        {
			Point p = vec.at(i);

			// sanity check
			if (!(rL==-1 && xscal==1) && (p.x > rL || p. y > rL || p.z > rL)) {   std::cout << "rL is too small \n"; exit (-1); }

			inputXHost[i] = p.x / xscal;
			inputYHost[i] = p.y / xscal;
			inputZHost[i] = p.z / xscal;
        }

        vec.clear();

        inputX = inputXHost;
        inputY = inputYHost;
        inputZ = inputZHost;

        return true;
    }


    // return a vector with N random values in the range [min,max)
    thrust::host_vector<float> random_vector(const size_t N, float max, float min)
    {
        thrust::host_vector<float> tmp(N);
        for(size_t i = 0; i < N; i++) tmp[i] = (u01(rng)*(max-min) + min);
        return tmp;
    }


    // clear vectors & variables
    void clear()
    {
        haloIndex.clear();
        haloIndexUnique.clear();

        haloColorsR.clear();
        haloColorsG.clear();
        haloColorsB.clear();

        numOfHalos = 0;
        numOfParticles = inputX.size();

        haloIndex.resize(numOfParticles);
        thrust::sequence(haloIndex.begin(), haloIndex.end());
    }


    // set colors to halos
    void setColors()
    {
        // set color range
        float minrangeC    = 0.1;
        float maxrangeC    = 1;

        // for each halo, set unique colors
        u01 = thrust::uniform_real_distribution<float>(0.0f, 1.0f);
        haloColorsR = random_vector(numOfHalos, maxrangeC, minrangeC);
        haloColorsG = random_vector(numOfHalos, maxrangeC, minrangeC);
        haloColorsB = random_vector(numOfHalos, maxrangeC, minrangeC);
    }


    // get unique halo ids & numOfHalos, Right now the halo id is not set to the smallest partucle id in the halo
    // there is a code you can comment out if you want to do that.
    void getUniqueHalos(int particleSize)
    {
        thrust::device_vector<int> haloUnique(numOfParticles);
        thrust::copy(haloIndex.begin(), haloIndex.end(), haloUnique.begin());
        thrust::sort(haloUnique.begin(), haloUnique.end()); //sort halo ids

        thrust::device_vector<int> haloSize(numOfParticles);
        thrust::fill(haloSize.begin(), haloSize.end(), 1);

        thrust::device_vector<int> a(numOfParticles);
        thrust::device_vector<int> b(numOfParticles);
 
        thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end; //get size of halo
        new_end = thrust::reduce_by_key(haloUnique.begin(), haloUnique.end(), haloSize.begin(), a.begin(), b.begin());

        int numUniqueHalos = thrust::get<0>(new_end) - a.begin();

        // get the number of valid halos & their ids
        thrust::device_vector<int>::iterator new_end1 = thrust::remove_if(a.begin(), thrust::get<0>(new_end), b.begin(), notValidHalo(particleSize));
        numOfHalos = new_end1 - a.begin();

        // for all particals in invalid halos, set halo id to -1
        thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfHalos,
		         resetInvalidParticles(thrust::raw_pointer_cast(&*a.begin()), numUniqueHalos, particleSize, thrust::raw_pointer_cast(&*haloIndex.begin())));

        haloIndexUnique = thrust::device_vector<int>(numOfHalos);
        thrust::copy(a.begin(), a.begin()+numOfHalos, haloIndexUnique.begin());

//        std::cout << "haloIndexUnique  "; thrust::copy(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfHalos, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;

        haloUnique.clear(); haloSize.clear(); a.clear(); b.clear();
    }


    struct is_NotEqual
    {
        int haloId;
        is_NotEqual(int haloId) : haloId(haloId) {}

        __host__ __device__
        bool operator()(const int x) { return haloId != x; }
    };

    //reset the halo id to -1, if a particle belongs to a invalid halo
    struct resetInvalidParticles
    {
        int *haloIndex;
        int *haloUnique;
        int numUniqueHalos, particleSize;

        resetInvalidParticles(int *haloUnique, int numUniqueHalos, int particleSize, int* haloIndex) :
	  		    haloUnique(haloUnique), numUniqueHalos(numUniqueHalos), particleSize(particleSize), haloIndex(haloIndex) {}

        __host__ __device__
        void operator()(int i)
        {
			bool found = false;
			for(int j=0; j<numUniqueHalos; j++) if(haloUnique[j] == haloIndex[i]) { found = true; break; }

			if(!found) haloIndex[i] = -1;
        }
    };

    // check whether number of particles in this halo exceed particleSize
    struct notValidHalo
    {
        int particleSize;
        notValidHalo(int particleSize) : particleSize(particleSize) {}

        __host__ __device__
        bool operator()(int i) { return i < particleSize; }
    };

    thrust::device_vector<int> getHalos()
	{
    	return haloIndex;
	}
};

}

#endif /* HALO_H_ */
