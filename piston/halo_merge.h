#ifndef HALO_MERGE_H
#define HALO_MERGE_H

#include <piston/halo.h>

#include "parallel_radix_sort.h"
#include <stdarg.h>

// When TEST is defined, output all results
//#define TEST

namespace piston
{

class halo_merge : public halo
{
public:
	struct Node
	{
		int   nodeId;// particle Id
    int   haloId;// halo Id, store the min particle id of the particles in this halo

		int   count; // store number of particles in the halo at this level
		float value; // weight value of this node

		Node *parent;       // pointer to my parent
		Node *parentSuper;  // pointer to my super parent

		__host__ __device__
		Node() { nodeId=-1; value=0.0f; haloId=-1; count=0;	parent=NULL; parentSuper=NULL; }

		__host__ __device__
		Node(int nodeId, float value, int haloId, int count, Node *parent, Node *parentSuper) :
			nodeId(nodeId), value(value), haloId(haloId), count(count), parent(parent), parentSuper(parentSuper) {}
	};

	struct Edge
	{
		int srcId, desId;
		float weight;

		__host__ __device__
		Edge() { srcId=-1; desId=-1; weight=-1; }
		
		__host__ __device__
		Edge(int srcId, int desId, float weight) : srcId(srcId), desId(desId), weight(weight) {}
	};

	float max_ll, min_ll;   // maximum & minimum linking length
	Point lBoundS, uBoundS; // lower & upper bounds of the entire space

	int mergetreeSize;
  int numOfEdges;
	int numOfCubes;					   	 					 // total number of cubes in space
	int cubesInX, cubesInY, cubesInZ;      // number of cubes in each dimension
	
  thrust::device_vector<int> particleId; // for each particle, particle id
  thrust::device_vector<int> cubeId;     // for each particle, cube id

	thrust::device_vector<int>   particleSizeOfCubes; 	// number of particles in cubes
	thrust::device_vector<int>   particleStartOfCubes;	// stratInd of cubes  (within particleId)

	thrust::device_vector<int>   neighborsOfCubes;			// neighbors of cubes (for each cube, store only 13 of them)

	thrust::device_vector<Node>  nodes;
  thrust::device_vector<Node>  nodesTmp1, nodesTmp2;

	thrust::device_vector<Edge>  edges;
	thrust::device_vector<int>   edgeSizeOfCubes;
	thrust::device_vector<int>   edgeStartOfCubes;
	
	thrust::device_vector<Edge>  tmpEdgeArray1;
	thrust::device_vector<float> tmpFloatArray1;
	thrust::device_vector<int>   tmpIntArray1, tmpIntArray2, tmpIntArray3;

	halo_merge(float min_linkLength, float max_linkLength, std::string filename="", std::string format=".cosmo",
						 int n = 1, int np=1, float rL=-1, bool periodic=false) : halo(filename, format, n, np, rL, periodic)
	{
		if(numOfParticles!=0)
		{
			struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3;

			//---- init stuff

			min_ll = min_linkLength; // get min_linkinglength
			max_ll = max_linkLength; // get max_linkinglength

			initDetails();

			//---- divide space into cubes
			gettimeofday(&begin, 0);
			divideIntoCubes();
			gettimeofday(&mid1, 0);

			std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;

			#ifdef TEST
				outputCubeDetails("init cube details"); // output cube details
			#endif

			//------- METHOD :
	    // parallel for each cube, get the set of edges, sort them & create the submerge tree.
			// globally combine them, two cubes at a time (by sorting their edge sets)

			gettimeofday(&mid2, 0);
			localStepMethod();
			gettimeofday(&mid3, 0);

			neighborsOfCubes.clear();

			std::cout << "-- localStep done" << std::endl;

			gettimeofday(&mid4, 0);
			globalStepMethod();
			gettimeofday(&end, 0);

			cubeId.clear();
			particleId.clear();
      particleSizeOfCubes.clear();
      particleStartOfCubes.clear();
			neighborsOfCubes.clear();

			edges.clear();
			edgeSizeOfCubes.clear();
			edgeStartOfCubes.clear();

			tmpEdgeArray1.clear();
			tmpFloatArray1.clear();
			tmpIntArray1.clear();
			tmpIntArray2.clear();
			tmpIntArray3.clear();

			std::cout << "-- globalStep done" << std::endl;

			timersub(&mid1, &begin, &diff1);
			float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
			std::cout << "Time elapsed: " << seconds1 << " s for dividing space into cubes"<< std::endl << std::flush;
			timersub(&mid3, &mid2, &diff2);
			float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
			std::cout << "Time elapsed: " << seconds2 << " s for localStep - finding edges in each cube"<< std::endl << std::flush;
			timersub(&end, &mid4, &diff3);
			float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
			std::cout << "Time elapsed: " << seconds3 << " s for globalStep - combine edges & creating the merge tree"<< std::endl << std::flush;
		}
	}

	void operator()(float linkLength, int  particleSize)
	{
		clear();

		linkLength   = linkLength;
		particleSize = particleSize;

		// no valid particles, return
		if(numOfParticles==0) return;

		struct timeval begin, end, diff;

		gettimeofday(&begin, 0);
		findHalos(linkLength, particleSize); // find halos
		gettimeofday(&end, 0);

		getSizeOfMeregeTree();
		getNumOfHalos();    // get the unique halo ids & set numOfHalos
		getHaloParticles(); // get the halo particles & set numOfHaloParticles
		setColors();        // set colors to halos

		timersub(&end, &begin, &diff);
		float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		std::cout << "Time elapsed: " << seconds << " s for finding halos at linking length " << linkLength << " and has particle size >= " << particleSize << std::endl << std::flush;
		std::cout << "Number of Particles : " << numOfParticles <<  " Number of Halos found : " << numOfHalos << std::endl << std::endl;
		std::cout << "mergetreeSize : " << mergetreeSize << std::endl << std::endl;
	}

	void findHalos(float linkLength, int particleSize)
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setHaloId(thrust::raw_pointer_cast(&*nodes.begin()),
						 		  thrust::raw_pointer_cast(&*haloIndex.begin()),
						      linkLength, particleSize));
	}

	// for a given node, set its halo id at the given link length, for particles in filtered halos (by particle size) id is set to -1
	struct setHaloId : public thrust::unary_function<int, void>
	{
		Node  *nodes;
		int   *haloIndex;
		int    particleSize;
		float  linkLength;

		__host__ __device__
		setHaloId(Node *nodes, int *haloIndex, float linkLength, int particleSize) :
			nodes(nodes), haloIndex(haloIndex), linkLength(linkLength), particleSize(particleSize) {}

		__host__ __device__
		void operator()(int i)
		{			
      Node n = ((Node)nodes[i]);

      if(n.parentSuper!=NULL && (n.parentSuper)->value<=linkLength)
        n = *(n.parentSuper);

      Node nChild;
      while(n.parent!=NULL && n.value < linkLength)
      {
        nChild = n;
        n = *(n.parent);
      }

      if(n.value >= linkLength)
        n = nChild;

      (nodes[i]).parentSuper = &n;

      haloIndex[i] = (n.count >= particleSize) ? n.haloId : -1;
		}
	};

	// get the unique halo indexes & number of halos
	void getNumOfHalos()
	{
	  thrust::device_vector<int>::iterator new_end;
		
		haloIndexUnique.resize(numOfParticles);		
		thrust::copy(haloIndex.begin(), haloIndex.end(), haloIndexUnique.begin());
		thrust::stable_sort(haloIndexUnique.begin(), haloIndexUnique.end());
	  new_end = thrust::unique(haloIndexUnique.begin(), haloIndexUnique.end());
	  new_end = thrust::remove(haloIndexUnique.begin(), new_end, -1);
	  numOfHalos = new_end - haloIndexUnique.begin();
	}

	// get the size of the merge tree
	void getSizeOfMeregeTree()
	{
		Node *nodesTmp = thrust::raw_pointer_cast(&*nodes.begin());

		mergetreeSize = 0;
		for(int i=0; i<numOfParticles; i++)
		{
			Node *n = &nodesTmp[i];

			while(n->parent!=NULL)
			{
				n = (n->parent);

				if(n->nodeId != -2) mergetreeSize++;
				n->nodeId = -2;								
			}
		}

		mergetreeSize += numOfParticles;
	}

	// get the particles of valid halos & get the number of halo particles *************************
	void getHaloParticles()
	{
	  thrust::device_vector<int>::iterator new_end;

	  tmpIntArray1.resize(numOfParticles);
	  new_end = thrust::remove_copy(haloIndex.begin(), haloIndex.end(), tmpIntArray1.begin(), -1);

	  numOfHaloParticles = new_end - tmpIntArray1.begin();

	  inputX_f.resize(numOfHaloParticles);
    inputY_f.resize(numOfHaloParticles);
    inputZ_f.resize(numOfHaloParticles);
	}



	//------- init stuff

	void initDetails()
	{
		initParticleIds();	// set particle ids
		getBounds();		    // get bounds of the entire space
		setNumberOfCubes();	// get total number of cubes
	}

  // set initial particle ids
	void initParticleIds()
	{
		particleId.resize(numOfParticles);
		thrust::sequence(particleId.begin(), particleId.end());
	}

	// get lower & upper bounds of the entire space
	void getBounds()
	{
		typedef thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> result_type;
		result_type result1 = thrust::minmax_element(inputX.begin(), inputX.end());
		result_type result2 = thrust::minmax_element(inputY.begin(), inputY.end());
		result_type result3 = thrust::minmax_element(inputZ.begin(), inputZ.end());

		lBoundS = Point(*result1.first,  *result2.first,  *result3.first);
		uBoundS = Point(*result1.second, *result2.second, *result3.second);
	}

	// get total number of cubes
	void setNumberOfCubes()
	{
		cubesInX = (std::ceil((uBoundS.x - lBoundS.x)/max_ll) == 0) ? 1 : std::ceil((uBoundS.x - lBoundS.x)/max_ll);
		cubesInY = (std::ceil((uBoundS.y - lBoundS.y)/max_ll) == 0) ? 1 : std::ceil((uBoundS.y - lBoundS.y)/max_ll);
		cubesInZ = (std::ceil((uBoundS.z - lBoundS.z)/max_ll) == 0) ? 1 : std::ceil((uBoundS.z - lBoundS.z)/max_ll);

		numOfCubes = cubesInX*cubesInY*cubesInZ;
	}



	//------- divide space into cubes

	void divideIntoCubes()
	{
	  struct timeval begin, mid1, mid2, mid3, end, diff0, diff1, diff2, diff3;
	  gettimeofday(&begin, 0);
		setCubeIds();		      		// for each particle, set cube id
	  gettimeofday(&mid1, 0);
		sortParticlesByCubeID();  // sort Particles by cube Id
	  gettimeofday(&mid2, 0);
		getSizeAndStartOfCubes(); // for each cube, count particles
	  gettimeofday(&mid3, 0);
    getNeighborsOfCubes();    // for each cube, get its neighbors
	  gettimeofday(&end, 0);

/*
    timersub(&mid1, &begin, &diff0);
    float seconds0 = diff0.tv_sec + 1.0E-6*diff0.tv_usec;
    std::cout << "Time elapsed0: " << seconds0 << " s for setCubeIds"<< std::endl << std::flush;
    timersub(&mid2, &mid1, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << " s for sortParticlesByCubeID"<< std::endl << std::flush;
    timersub(&mid3, &mid2, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed3: " << seconds2 << " s for getSizeAndStartOfCubes"<< std::endl << std::flush;
    timersub(&end, &mid3, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed4: " << seconds3 << " s for getNeighborsOfCubes "<< std::endl << std::flush;
*/
	}

	// set cube ids of particles
	void setCubeIds()
	{
	  cubeId.resize(numOfParticles);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setCubeIdOfParticle(thrust::raw_pointer_cast(&*inputX.begin()),
														thrust::raw_pointer_cast(&*inputY.begin()),
														thrust::raw_pointer_cast(&*inputZ.begin()),
														thrust::raw_pointer_cast(&*cubeId.begin()),
														max_ll, lBoundS, cubesInX, cubesInY, cubesInZ));
	}

	// for a given particle, set its cube id
	struct setCubeIdOfParticle : public thrust::unary_function<int, void>
	{
		float  max_ll;

		Point  lBoundS;
		int    cubesInX, cubesInY, cubesInZ;

		int   *cubeId;
		float *inputX, *inputY, *inputZ;

		__host__ __device__
		setCubeIdOfParticle(float *inputX, float *inputY, float *inputZ,
			int *cubeId, float max_ll, Point lBoundS,
			int cubesInX, int cubesInY, int cubesInZ) :
			inputX(inputX), inputY(inputY), inputZ(inputZ),
			cubeId(cubeId), max_ll(max_ll), lBoundS(lBoundS),
			cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ){}

		__host__ __device__
		void operator()(int i)
		{
			// get x,y,z coordinates for the cube
			int z = (((inputZ[i]-lBoundS.z) / max_ll)>=cubesInZ) ? cubesInZ-1 : (inputZ[i]-lBoundS.z) / max_ll;
			int y = (((inputY[i]-lBoundS.y) / max_ll)>=cubesInY) ? cubesInY-1 : (inputY[i]-lBoundS.y) / max_ll;
			int x = (((inputX[i]-lBoundS.x) / max_ll)>=cubesInX) ? cubesInX-1 : (inputX[i]-lBoundS.x) / max_ll;

			// get cube id
			cubeId[i] = (z*(cubesInX*cubesInY) + y*cubesInX + x);
		}
	};

	// sort particles by cube id
	void sortParticlesByCubeID()
	{
    thrust::stable_sort_by_key(cubeId.begin(), cubeId.end(), particleId.begin());
	}
		
	// for each cube, get the size & start of cube particles (in particleId array)
	void getSizeAndStartOfCubes()
	{
		tmpIntArray1.resize(numOfCubes);
		tmpIntArray2.resize(numOfCubes);

		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
		new_end = thrust::reduce_by_key(cubeId.begin(), cubeId.end(), thrust::constant_iterator<int>(1), tmpIntArray1.begin(), tmpIntArray2.begin());

		int size = (thrust::get<0>(new_end)-tmpIntArray1.begin());

		particleSizeOfCubes.resize(numOfCubes);
		thrust::scatter(tmpIntArray2.begin(), tmpIntArray2.begin()+size, tmpIntArray1.begin(), particleSizeOfCubes.begin());

		particleStartOfCubes.resize(numOfCubes);
		thrust::exclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, particleStartOfCubes.begin());

		tmpIntArray1.clear();
		tmpIntArray2.clear();
	}
	
	// for each cube, get its neighbors, here we only set 13 of them not all 26
	void getNeighborsOfCubes()
	{
		neighborsOfCubes.resize(13*numOfCubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				setNeighboringCubes(cubesInX, cubesInY, cubesInZ,
														thrust::raw_pointer_cast(&*neighborsOfCubes.begin())));
	}

	// for a given cube where center is (x,y,z), get its neighboring cubes
	struct setNeighboringCubes : public thrust::unary_function<int, void>
	{
		int  cubesInX, cubesInY, cubesInZ;
		int *neighborsOfCubes;

		__host__ __device__
		setNeighboringCubes(int cubesInX, int cubesInY, int cubesInZ, int *neighborsOfCubes) :
				cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), neighborsOfCubes(neighborsOfCubes) {}

		__host__ __device__
		void operator()(int i)
		{
			int startInd = i*13;

			int tmp = i % (cubesInX*cubesInY);

			// get x,y,z coordinates for the cube
			int z = i / (cubesInX*cubesInY);
			int y = tmp / cubesInX;
			int x = tmp % cubesInX;
	
			neighborsOfCubes[startInd+0]  = (x+1<cubesInX) ? (z*(cubesInX*cubesInY) + y*cubesInX + (x+1)) : -1; 																				// x+1,y,z
			neighborsOfCubes[startInd+1]  = (y+1<cubesInY) ? (z*(cubesInX*cubesInY) + (y+1)*cubesInX + x) : -1;	 																				// x,y+1,z
			neighborsOfCubes[startInd+2]  = (z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + y*cubesInX + x) : -1;  																				// x,y,z+1

			neighborsOfCubes[startInd+3]  = (x+1<cubesInX && y+1<cubesInY) ? (z*(cubesInX*cubesInY) + (y+1)*cubesInX + (x+1)) : -1; 										// x+1,y+1,z
			neighborsOfCubes[startInd+4]  = (x+1<cubesInX && y-1>=0) ? (z*(cubesInX*cubesInY) + (y-1)*cubesInX + (x+1)) : -1;		      									// x+1,y-1,z

			neighborsOfCubes[startInd+5]  = (x+1<cubesInX && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + y*cubesInX + (x+1)) : -1; 										// x+1,y,z+1
			neighborsOfCubes[startInd+6]  = (x+1<cubesInX && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + y*cubesInX + (x+1)) : -1;		     										// x+1,y,z-1

			neighborsOfCubes[startInd+7]  = (y+1<cubesInY && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + (y+1)*cubesInX + x) : -1;	 										// x,y+1,z+1
			neighborsOfCubes[startInd+8]  = (y+1<cubesInY && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + (y+1)*cubesInX + x) : -1;			   										// x,y+1,z-1

			neighborsOfCubes[startInd+9]  = (x+1<cubesInX && y+1<cubesInY && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + (y+1)*cubesInX + (x+1)) : -1; // x+1,y+1,z+1
			neighborsOfCubes[startInd+10] = (x+1<cubesInX && y+1<cubesInY && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + (y+1)*cubesInX + (x+1)) : -1;		    // x+1,y+1,z-1

			neighborsOfCubes[startInd+11] = (x+1<cubesInX && y-1>=0 && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + (y-1)*cubesInX + (x+1)) : -1; 			// x+1,y-1,z+1
			neighborsOfCubes[startInd+12] = (x+1<cubesInX && y-1>=0 && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + (y-1)*cubesInX + (x+1)) : -1;		    		  // x+1,y-1,z-1			
		}
	};



	//------- output results

	// print cube details from device vectors
	void outputCubeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << std::endl << "-- Outputs------------" << std::endl << std::endl;

		std::cout << "-- Dim    (" << lBoundS.x << "," << lBoundS.y << "," << lBoundS.z << "), (";
		std::cout << uBoundS.x << "," << uBoundS.y << "," << uBoundS.z << ")" << std::endl;
		std::cout << "-- max_ll " << max_ll << std::endl;
		std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;

		std::cout << std::endl << "----------------------" << std::endl << std::endl;

		std::cout << "particleId 	"; thrust::copy(particleId.begin(), particleId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "cubeID		"; thrust::copy(cubeId.begin(), cubeId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputX		"; thrust::copy(inputX.begin(), inputX.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputY		"; thrust::copy(inputY.begin(), inputY.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputZ		"; thrust::copy(inputZ.begin(), inputZ.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "startOfCube	"; thrust::copy(particleStartOfCubes.begin(), particleStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "neighborsOfCubes	"; thrust::copy(neighborsOfCubes.begin(), neighborsOfCubes.begin()+13*numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;

		std::cout << "----------------------" << std::endl << std::endl;
	}

	// print edge details from device vectors
	void outputEdgeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << "numOfEdges			 " << numOfEdges << std::endl << std::endl;
		std::cout << "edgeSizeOfCubes	 "; thrust::copy(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "edgeStartOfCubes "; thrust::copy(edgeStartOfCubes.begin(), edgeStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << std::endl;

		for(int i=0; i<numOfEdges; i++)
			std::cout << "---- " << ((Edge)edges[i]).srcId << "," << ((Edge)edges[i]).desId << "," << ((Edge)edges[i]).weight <<  ")" << std::endl;
		std::cout << std::endl;

		std::cout << "----------------------" << std::endl << std::endl;
	}
	
	// print merge tree details from device vectors
	void outputMergeTreeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << "MergeTreeNodes " << std::endl;
		for(int i=0; i<numOfParticles; i++)
		{
			Node n = ((Node)nodes[i]);
			int k  = (n.parent==NULL)?0:1;
			std::cout << "(" << inputX[i] << "," << inputY[i] << "," << inputZ[i] << " : " << n.value << "," << n.nodeId << "," << n.haloId << "," << k << ")";

			while(n.parent!=NULL)
			{
				n = *(n.parent);
				k = (n.parent==NULL)?0:1;
				std::cout << "(" << n.value << "," << n.nodeId << "," << n.haloId << "," << k << ")";
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << "----------------------" << std::endl << std::endl;
	}



	//------- METHOD : parallel for each cube, get the set of edges & create the submerge tree. Globally combine them

	//---------------- METHOD - Local functions

	// locally, get the set of edges for each cube & create the submerge tree
	void localStepMethod()
	{
	  struct timeval begin, mid1, mid2, mid3, mid4, end, diff0, diff1, diff2, diff3, diff4;
	  gettimeofday(&begin, 0);
		initEdgeArrays();  				// init arrays needed for storing edges
		gettimeofday(&mid1, 0);
		getEdgesPerCube(); 				// for each cube, get the set of edges
		gettimeofday(&mid2, 0);
		sortEdgesPerCube(); 			// for each cube, sort the set of edges by weight
		gettimeofday(&mid3, 0);
		sortCubeIDByParticleID(); // sort cube ids by particle ids
		gettimeofday(&mid4, 0);
		getSubMergeTreePerCube(); // for each cube, get the sub merge tree
		gettimeofday(&end, 0);
		#ifdef TEST
			outputEdgeDetails("After removing unecessary edges found in each cube..");	  // output edge details
			outputMergeTreeDetails("The sub merge trees.."); // output merge tree details
		#endif

    timersub(&mid1, &begin, &diff0);
    float seconds0 = diff0.tv_sec + 1.0E-6*diff0.tv_usec;
    std::cout << "Time elapsed0: " << seconds0 << " s for initEdgeArrays"<< std::endl << std::flush;
    timersub(&mid2, &mid1, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << " s for getEdgesPerCube"<< std::endl << std::flush;
    timersub(&mid3, &mid2, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed3: " << seconds2 << " s for sortEdgesPerCube"<< std::endl << std::flush;
    timersub(&mid4, &mid3, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed4: " << seconds3 << " s for sortCubeIDByParticleID "<< std::endl << std::flush;
    timersub(&end, &mid4, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed5: " << seconds4 << " s for getSubMergeTreePerCube"<< std::endl << std::flush;
	}

	// for each cube, init arrays needed for storing edges
	void initEdgeArrays()
	{
		// for each cube, set the space required for storing edges
		edgeSizeOfCubes.resize(numOfCubes);

		for(int i=0; i<=13; i++)
		{
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
					getEdgesSizes(thrust::raw_pointer_cast(&*neighborsOfCubes.begin()),
									      thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
									      thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
									      thrust::raw_pointer_cast(&*inputX.begin()),
									      thrust::raw_pointer_cast(&*inputY.begin()),
									      thrust::raw_pointer_cast(&*inputZ.begin()),
									      thrust::raw_pointer_cast(&*particleId.begin()),
									      max_ll, min_ll, i,
									      thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin())));
		}
		edgeStartOfCubes.resize(numOfCubes);
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes, edgeStartOfCubes.begin());

		numOfEdges = edgeStartOfCubes[numOfCubes-1] + edgeSizeOfCubes[numOfCubes-1];

		// init edge arrays
		edges.resize(numOfEdges);
	}

  // for each cube, get the size of edges for allocations of edge array
  struct getEdgesSizes : public thrust::unary_function<int, void>
  {
    int    num;
    float  max_ll, min_ll;
    float *inputX, *inputY, *inputZ;

    int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;
    int   *neighborsOfCubes;

    int  *edgeSizeOfCubes;

    __host__ __device__
    getEdgesSizes(int *neighborsOfCubes, int *particleStartOfCubes, int *particleSizeOfCubes,
        float *inputX, float *inputY, float *inputZ,
        int *particleId, float max_ll, float min_ll, int num, int *edgeSizeOfCubes) :
        neighborsOfCubes(neighborsOfCubes),
        particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
        inputX(inputX), inputY(inputY), inputZ(inputZ),
        particleId(particleId), max_ll(max_ll), min_ll(min_ll), num(num),
        edgeSizeOfCubes(edgeSizeOfCubes) {}

     __host__ __device__
     void operator()(int i)
     {
       int cube = (num==0) ? i : neighborsOfCubes[13*i+(num-1)];

       if(cube==-1 || particleSizeOfCubes[i]==0 || particleSizeOfCubes[cube]==0) return;

       // for each particle in cube
       int size  = 0;
       for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
       {
          float currentX = inputX[particleId[j]];
          float currentY = inputY[particleId[j]];
          float currentZ = inputZ[particleId[j]];

          // compare with particles in this cube
          int start = (num==0) ? j+1 : particleStartOfCubes[cube];
          for(int k=start; k<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; k++)
          {
            float otherX = inputX[particleId[k]];
            float otherY = inputY[particleId[k]];
            float otherZ = inputZ[particleId[k]];

            float xd, yd, zd;
            xd = (currentX-otherX);  if (xd < 0.0f) xd = -xd;
            yd = (currentY-otherY);  if (yd < 0.0f) yd = -yd;
            zd = (currentZ-otherZ);  if (zd < 0.0f) zd = -zd;

            if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
            {
              float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);

              if(dist <= max_ll) size++;
            }
         }
       }

       edgeSizeOfCubes[i] += size;
    }
  };

	// for each cube, get the set of edges
	void getEdgesPerCube()
	{	
		for(int i=0; i<=13; i++)
		{
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
					getEdges(thrust::raw_pointer_cast(&*neighborsOfCubes.begin()), 
									 thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
									 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
									 thrust::raw_pointer_cast(&*inputX.begin()),
									 thrust::raw_pointer_cast(&*inputY.begin()),
									 thrust::raw_pointer_cast(&*inputZ.begin()),
									 thrust::raw_pointer_cast(&*particleId.begin()),
									 max_ll, i,
									 thrust::raw_pointer_cast(&*edges.begin()),
									 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
									 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));			
		}
	}

	// for each cube, get the set of edges after comparing
	struct getEdges : public thrust::unary_function<int, void>
	{
		int    num;
		float  max_ll;
		float *inputX, *inputY, *inputZ;

		int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;
		int   *neighborsOfCubes;
		
		Edge *edges;
		int  *edgeStartOfCubes, *edgeSizeOfCubes;

		__host__ __device__
		getEdges(int *neighborsOfCubes, int *particleStartOfCubes, int *particleSizeOfCubes, 
				float *inputX, float *inputY, float *inputZ,
				int *particleId, float max_ll, int num,
				Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes) :
				neighborsOfCubes(neighborsOfCubes), 
				particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
				inputX(inputX), inputY(inputY), inputZ(inputZ),
				particleId(particleId), max_ll(max_ll), num(num),
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

		 __host__ __device__
     void operator()(int i)
		 {
			 int cube = i;
			 if(num==0)
				 edgeSizeOfCubes[i] = 0;
			 else
				 cube = neighborsOfCubes[13*i+(num-1)];

			 if(cube==-1 || particleSizeOfCubes[i]==0 || particleSizeOfCubes[cube]==0) return;

			 // for each particle in cube
			 for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
			 {
					float currentX = inputX[particleId[j]];
				  float currentY = inputY[particleId[j]];
				  float currentZ = inputZ[particleId[j]];

				  // compare with particles in this cube
				  int start = (num==0) ? j+1 : particleStartOfCubes[cube];
				  for(int k=start; k<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; k++)
				  {
						float otherX = inputX[particleId[k]];
						float otherY = inputY[particleId[k]];
						float otherZ = inputZ[particleId[k]];

						float xd, yd, zd;
						xd = (currentX-otherX);  if (xd < 0.0f) xd = -xd;
						yd = (currentY-otherY);  if (yd < 0.0f) yd = -yd;
						zd = (currentZ-otherZ);  if (zd < 0.0f) zd = -zd;
												                                                                                                                                       						
						if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
						{
							float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);
							if(dist <= max_ll)
							{
								int src = (particleId[j] <= particleId[k]) ? particleId[j] : particleId[k];
								int des = (src == particleId[k]) ? particleId[j] : particleId[k];
															                                                                                                                                                 
								// add edge
								edges[edgeStartOfCubes[i] + edgeSizeOfCubes[i]] = Edge(src, des, dist);
							
								edgeSizeOfCubes[i]++;
							}
						}
				 }				
			}
		}
	};

void InitRandom(int *a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = rand();
  }
}

  // for each cube, sort the set of edges by weight
  void sortEdgesPerCube()
  {
    tmpIntArray1.resize(numOfEdges); // stores the keys

    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
        setValuesAndKeys(thrust::raw_pointer_cast(&*edges.begin()),
                 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
                 max_ll,
                 thrust::raw_pointer_cast(&*tmpIntArray1.begin())));
//--------------
/*
		std::cout << numOfEdges << std::endl;

		int N = numOfEdges;

		int *keys, *vals;
		keys = new int[N];
		vals = new int[N];
		InitRandom(keys, N);
		InitRandom(vals, N);
//	  int  *keys = thrust::raw_pointer_cast(&*tmpIntArray1.begin());
//	  Edge *vals = thrust::raw_pointer_cast(&*edges.begin());

		for(size_t i = 0; i < N; i+=N/20){
			std::cout << keys[i] << " ";
		}
		std::cout << std::endl;

		std::cout << N << std::endl;

	  struct timeval begin, end, diff;
	  gettimeofday(&begin, 0);
		parallel_radix_sort::SortPairs(keys, vals, N);
	  gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "Time elapsed: " << seconds << " parallel_radix_sort" << std::endl << std::flush;
*/    
//--------------

    thrust::stable_sort_by_key(tmpIntArray1.begin(), tmpIntArray1.end(), edges.begin());

    tmpIntArray1.clear();
  }

  // set values & keys arrays needed for edge sorting
  struct setValuesAndKeys : public thrust::unary_function<int, void>
  {
    float max_ll;

    int *keys;

    Edge *edges;
    int  *edgeSizeOfCubes, *edgeStartOfCubes;

    __host__ __device__
    setValuesAndKeys(Edge *edges, int *edgeSizeOfCubes,int *edgeStartOfCubes,
        float max_ll, int *keys) :
        edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
        max_ll(max_ll), keys(keys) {}

    __host__ __device__
    void operator()(int i)
    {
      for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
      {
        keys[j] = (edges[j].weight + 2*i*max_ll)*1000; //(float) (edges[j].weight + 2*i*max_ll);
      }
    }
  };

  // sort cube id by particle id
  void sortCubeIDByParticleID()
  {
    thrust::device_vector<int> temp(particleId.begin(), particleId.end());
    thrust::stable_sort_by_key(temp.begin(), temp.end(), cubeId.begin());

//    thrust::sort_by_key(particleId.begin(), particleId.end(), cubeId.begin());
  }

	// for each cube, compute the sub merge trees
	void getSubMergeTreePerCube(bool consider=false)
	{
		// clear stuff
		nodes.clear();
		nodesTmp1.clear();
		nodesTmp2.clear();

		nodes.resize(numOfParticles);
		nodesTmp1.resize(numOfEdges);
		nodesTmp2.resize(numOfEdges); 

		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				initNodes(thrust::raw_pointer_cast(&*nodes.begin())));

		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				getSubMergeTree(thrust::raw_pointer_cast(&*edges.begin()),
				                thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
												thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
												thrust::raw_pointer_cast(&*particleId.begin()),
												thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
												thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
												thrust::raw_pointer_cast(&*cubeId.begin()),
												thrust::raw_pointer_cast(&*nodesTmp1.begin()),
												thrust::raw_pointer_cast(&*nodesTmp2.begin()),
												thrust::raw_pointer_cast(&*nodes.begin()),
												min_ll, consider));

    removeEmptyEdges(); // remove empty items in edge sets
	}

	// init the nodes array with node id & halo id
  struct initNodes
  {
		Node *nodes;

		__host__ __device__
		initNodes(Node * nodes) : nodes(nodes) {}

    __host__ __device__
    void operator()(int i)
    {
			nodes[i].nodeId = i;
    	nodes[i].haloId = i;
			nodes[i].count  = 1;
    }
 	};

  // for a given cube, compute its sub merge tree
  struct getSubMergeTree : public thrust::unary_function<int, void>
  {
    float min_ll;
    bool  considerTmp;

    Edge *edges;
    int  *edgeStartOfCubes, *edgeSizeOfCubes;

    int  *particleId;
    int  *particleSizeOfCubes, *particleStartOfCubes;
    int  *cubeId;

    Node *nodes, *nodesTmp1, *nodesTmp2;

    __host__ __device__
    getSubMergeTree(Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes,
        int *particleId, int *particleSizeOfCubes, int *particleStartOfCubes,
        int *cubeId, Node *nodesTmp1, Node *nodesTmp2, Node *nodes, float min_ll, bool considerTmp) :
        edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
        particleId(particleId), particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes),
        cubeId(cubeId), nodesTmp1(nodesTmp1), nodesTmp2(nodesTmp2), nodes(nodes), min_ll(min_ll), considerTmp(considerTmp) {}

    __host__ __device__
    void operator()(int i)
    {
      int size=0, nxt1=0;

      bool adjusted = false;

      for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
      {
        Edge e = ((Edge)edges[j]);

        if(!adjusted && e.weight >= min_ll)
        {
          for(int k = particleStartOfCubes[i]; k<particleStartOfCubes[i]+particleSizeOfCubes[i]; k++)
          {
            Node *a = &nodes[particleId[k]];

						if(a->parent==NULL) continue;

            Node *b = (a->parentSuper==NULL) ? a : a->parentSuper;
            while(b->parent!=NULL) b = (b->parentSuper==NULL) ? b->parent : b->parentSuper;

            a->parent      = b;
            a->parentSuper = b;
          }

          for(int k = edgeStartOfCubes[i]; k<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; k++)
          {
            Node *a = &nodesTmp2[k];

            if(a->nodeId==-1)   break;
						if(a->parent==NULL) continue;

            Node *b = (a->parentSuper==NULL) ? a : a->parentSuper;
            while(b->parent!=NULL) b = (b->parentSuper==NULL) ? b->parent : b->parentSuper;

            a->parent      = b;
            a->parentSuper = b;
          }

          adjusted = true;
        }

        // get the two merge tree leaf nodes to compare
        Node *tmp1, *tmp2, *tmp3, *tmp4;

        tmp1 = &nodes[e.srcId];
        tmp2 = &nodes[e.desId];

        bool setTmp1=(cubeId[e.srcId]==i), setTmp2=(cubeId[e.desId]==i);

        if(!setTmp1 || !setTmp2)
        {
          for(int k = edgeStartOfCubes[i]; k<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; k++)
          {
            if(!setTmp1 && (nodesTmp2[k].nodeId==e.srcId || nodesTmp2[k].nodeId==-1))
            {
              setTmp1 = true;
              tmp1    = &nodesTmp2[k];
              tmp1->nodeId = e.srcId;
              tmp1->haloId = e.srcId;
              continue;
            }

            if(!setTmp2 && (nodesTmp2[k].nodeId==e.desId || nodesTmp2[k].nodeId==-1))
            {
              setTmp2 = true;
              tmp2    = &nodesTmp2[k];
              tmp2->nodeId = e.desId;
              tmp2->haloId = e.desId;
              continue;
            }

            if(setTmp1 && setTmp2) break;
          }
        }

        // get the superparents of the two leafs
        tmp3 = (tmp1->parentSuper==NULL) ? tmp1 : tmp1->parentSuper;
        tmp4 = (tmp2->parentSuper==NULL) ? tmp2 : tmp2->parentSuper;

        while(tmp3->parent!=NULL)  tmp3 = (tmp3->parentSuper==NULL) ? tmp3->parent : tmp3->parentSuper;
        while(tmp4->parent!=NULL)  tmp4 = (tmp4->parentSuper==NULL) ? tmp4->parent : tmp4->parentSuper;

        // set the superparent of the two leafs
        tmp1->parentSuper = tmp3;
        tmp2->parentSuper = tmp4;

        // if haloIds are different, connect them
        if(tmp3->haloId != tmp4->haloId)
        {
          edges[edgeStartOfCubes[i]+size] = e;
          size++;

          int minValue = (tmp3->haloId < tmp4->haloId) ? tmp3->haloId : tmp4->haloId;

					if(tmp3->value == e.weight || tmp4->value == e.weight)
          {
            Node *n1=tmp4, *n2=tmp3, *n3=tmp1;
            if(tmp3->value == e.weight)
            {
              n1=tmp3;  n2=tmp4;  n3=tmp2;
            }

            n1->haloId      = minValue;
            n1->count      += n2->count;

            n2->parent      = n1;
            n3->parentSuper = n1;

            if(e.weight < min_ll) n3->parent = n1;
          }
          else
          {
            Node *n   = &nodesTmp1[edgeStartOfCubes[i]+nxt1];
            n->value  = e.weight;
            n->haloId = minValue;
            n->count += (tmp3->count + tmp4->count);

            tmp3->parent      = n;
            tmp4->parent      = n;
            tmp1->parentSuper = n;
            tmp2->parentSuper = n;

            if(e.weight < min_ll)
            {
              tmp1->parent = n;   tmp2->parent = n;
            }

            nxt1++;
          }
        }
      }

      edgeSizeOfCubes[i] = size;
    }
 };

	// remove the empty edges in th edges array
  void removeEmptyEdges()
  {
    // get new number of edges
    numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes);
    tmpEdgeArray1.resize(numOfEdges);

    int size = 0;
    for(int i=0; i<numOfCubes; i++) // copy the new set of edges, per cube
    {
      int a = edgeStartOfCubes[i];
      int b = edgeSizeOfCubes[i];
      thrust::copy(edges.begin()+a,
                   edges.begin()+a+b,
                   tmpEdgeArray1.begin()+size);

      edgeStartOfCubes[i] = size;
      size += b; // for each cube, get new start of edges
    }
    edges = tmpEdgeArray1;

    tmpEdgeArray1.clear();
  }

	//---------------- METHOD - Global functions

	// globally, combine the sub merge trees by combining two cubes at a time
	void globalStepMethod()
	{
		// set new number of cubes
		int numOfCubesOld = numOfCubes;
		numOfCubes = (int)std::ceil(((double)numOfCubes/2));

		if(numOfEdges==0) return;

		while(numOfCubes!=numOfCubesOld && numOfCubes>0)
		{
			tmpIntArray1.resize(2*numOfCubes); // store the combined cube ids
			thrust::merge(CountingIterator(0), CountingIterator(0)+numOfCubes, CountingIterator(0), CountingIterator(0)+numOfCubes, tmpIntArray1.begin());

			// set new cube Ids
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles, 
					setNewCubeID(thrust::raw_pointer_cast(&*cubeId.begin())));

			tmpIntArray2.resize(numOfCubes);
			tmpIntArray3.resize(numOfCubes);

      //set new particleSizeOfCubes
      thrust::reduce_by_key(tmpIntArray1.begin(), tmpIntArray1.begin()+numOfCubesOld, particleSizeOfCubes.begin(), tmpIntArray2.begin(), tmpIntArray3.begin());
      particleSizeOfCubes = tmpIntArray3;

      // set new particleStartOfCubes
      thrust::unique_by_key_copy(tmpIntArray1.begin(), tmpIntArray1.begin()+numOfCubesOld, particleStartOfCubes.begin(), tmpIntArray2.begin(), tmpIntArray3.begin());
      particleStartOfCubes = tmpIntArray3;

      #ifdef TEST
        outputCubeDetails("The new cube details.."); // output cube details
      #endif

			//set new sizeOfEdges
			thrust::reduce_by_key(tmpIntArray1.begin(), tmpIntArray1.begin()+numOfCubesOld, edgeSizeOfCubes.begin(), tmpIntArray2.begin(), tmpIntArray3.begin());
			edgeSizeOfCubes = tmpIntArray3;

			// set new startOfEdges
			thrust::unique_by_key_copy(tmpIntArray1.begin(), tmpIntArray1.begin()+numOfCubesOld, edgeStartOfCubes.begin(), tmpIntArray2.begin(), tmpIntArray3.begin());
			edgeStartOfCubes = tmpIntArray3;

			#ifdef TEST
				outputEdgeDetails("The edge details for new cubes. ."); // output edge details
			#endif

			// for each new cube, sort new sets of edges
			sortEdgesPerCube(); 			

			#ifdef TEST
				outputEdgeDetails("After sorting edges for new cubes. ."); // output edge details
			#endif

			// for each new cube, get the sub merge tree
			getSubMergeTreePerCube(true);

			#ifdef TEST
				outputMergeTreeDetails("The new sub merge trees.."); // output merge tree details
			#endif

			// set new number of cubes
			numOfCubesOld = numOfCubes;
			numOfCubes = (int)std::ceil(((double)numOfCubes/2));

			tmpIntArray1.clear();
			tmpIntArray2.clear();
			tmpIntArray3.clear();
		}
	}

	// for each cube, set new cube Ids
	struct setNewCubeID  : public thrust::unary_function<int, void>
	{
		int *cubeId;

		__host__ __device__
		setNewCubeID(int *cubeId) : cubeId(cubeId) {}

		__host__ __device__
		void operator()(int i)
		{
			cubeId[i] = (cubeId[i]%2 == 0) ? cubeId[i]/2 : (cubeId[i]-1)/2;
		}
	};

};

}

#endif
