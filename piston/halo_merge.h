#ifndef HALO_MERGE_H
#define HALO_MERGE_H

#include <piston/halo.h>

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

		Node *childE, *childS; 
		Node *sibling;       

		__host__ __device__
		Node() { nodeId=-1; value=0.0f; haloId=-1; count=0;	parent=NULL; parentSuper=NULL; childS=NULL; childE=NULL; sibling=NULL; }

		__host__ __device__
		Node(int nodeId, float value, int haloId, int count, Node *parent, Node *parentSuper) :
			nodeId(nodeId), value(value), haloId(haloId), count(count), parent(parent), parentSuper(parentSuper) {}
	};

	struct Edge
	{
		int srcId, desId, tmp;
		float weight;

		__host__ __device__
		Edge() { srcId=-1; desId=-1; weight=-1; tmp=-1; }
		
		__host__ __device__
		Edge(int srcId, int desId, float weight, int tmp=-1) : srcId(srcId), desId(desId), weight(weight), tmp(tmp) {}
	};

	float cubeLen;					// length of the cube
	float max_ll, min_ll;   // maximum & minimum linking length
	Point lBoundS, uBoundS; // lower & upper bounds of the entire space

	float totalTime; // total time taken fopr execution

	int mergetreeSize;                // total size of the global merge tree
  int numOfEdges;                   // total number of edges in space
	int numOfCubes;					   	 			// total number of cubes in space
	int cubesInX, cubesInY, cubesInZ; // number of cubes in each dimension
	
  thrust::device_vector<int>   particleId; // for each particle, particle id
  thrust::device_vector<int>   cubeId;     // for each particle, cube id

	thrust::device_vector<int>   particleSizeOfCubes; 	// number of particles in cubes
	thrust::device_vector<int>   particleStartOfCubes;	// stratInd of cubes  (within particleId)

	thrust::device_vector<int>   neighborsOfCubes;			// neighbors of cubes (for each cube, store only 13 of them)

	thrust::device_vector<Node>  nodes;
  thrust::device_vector<Node>  nodesTmp1;

	thrust::device_vector<Edge>  edges;
	thrust::device_vector<int>   edgeSizeOfCubes;
	thrust::device_vector<int>   edgeStartOfCubes;
	
	thrust::device_vector<double> tmpDoubleArray1;
	thrust::device_vector<int>    tmpIntArray1, tmpIntArray2, tmpNxt, tmpFree;

	halo_merge(float min_linkLength, float max_linkLength, std::string filename="", std::string format=".cosmo",
						 int n = 1, int np=1, float rL=-1, bool periodic=false) : halo(filename, format, n, np, rL, periodic)
	{
		if(numOfParticles!=0)
		{
			struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3;

			//---- init stuff
			min_ll = min_linkLength; // get min_linkinglength
			max_ll = max_linkLength; // get max_linkinglength

			cubeLen = std::sqrt((min_ll*min_ll)/2); // min_ll^2 = cubeLen^2 + cubeLen^2	

			if(cubeLen <= 0) { std::cout << "--ERROR : plase specify a valid cubeLen... current cubeLen is " << cubeLen << std::endl; return; }

			initDetails();

			//---- divide space into cubes
			gettimeofday(&begin, 0);
			divideIntoCubes();
			gettimeofday(&mid1, 0);

			std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ") ... cubeLen " << cubeLen << std::endl;

			#ifdef TEST
				outputCubeDetails("init cube details"); // output cube details
			#endif

			//------- METHOD :
	    // parallel for each cube, get the set of edges, sort them & create the submerge tree.
			// globally combine the local merge trees, two cubes at a time

			gettimeofday(&mid2, 0);
			localStep();
			gettimeofday(&mid3, 0);

			std::cout << "-- localStep done" << std::endl;

			neighborsOfCubes.clear();

			gettimeofday(&mid4, 0);
			globalStep();
			gettimeofday(&end, 0);

			std::cout << "-- globalStep done" << std::endl;

			checkValidMergeTree();

			clearSuperParents();

			cubeId.clear();
			particleId.clear();
      particleSizeOfCubes.clear();
      particleStartOfCubes.clear();

			edges.clear();
			edgeSizeOfCubes.clear();
			edgeStartOfCubes.clear();

			tmpDoubleArray1.clear();
			tmpIntArray1.clear();
			tmpIntArray2.clear();

			tmpNxt.clear();
			tmpFree.clear();

			std::cout << std::endl;
			timersub(&mid1, &begin, &diff1);
			float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
			std::cout << "Time elapsed: " << seconds1 << " s for dividing space into cubes"<< std::endl << std::flush;
			timersub(&mid3, &mid2, &diff2);
			float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
			std::cout << "Time elapsed: " << seconds2 << " s for localStep - finding edges in each cube"<< std::endl << std::flush;
			timersub(&end, &mid4, &diff3);
			float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
			std::cout << "Time elapsed: " << seconds3 << " s for globalStep - combine edges & creating the merge tree"<< std::endl << std::flush;
			totalTime = seconds1 + seconds2 + seconds3;
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

		timersub(&end, &begin, &diff);
		float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		totalTime +=seconds;

		std::cout << "Time elapsed: " << seconds << " s for finding halos at linking length " << linkLength << " and has particle size >= " << particleSize << std::endl << std::flush;
		std::cout << "Total time elapsed: " << totalTime << " s" << std::endl << std::endl;

		getSizeOfMeregeTree();
		getNumOfHalos();    // get the unique halo ids & set numOfHalos
		getHaloParticles(); // get the halo particles & set numOfHaloParticles
		setColors();        // set colors to halos

		std::cout << "Number of Particles : " << numOfParticles << std::endl;
		std::cout << "Number of Halos found : " << numOfHalos << std::endl << std::endl;
		std::cout << "Merge tree size : " << mergetreeSize << std::endl;
    std::cout << "Min_ll  : " << min_ll  << std::endl;
    std::cout << "Max_ll  : " << max_ll << std::endl << std::endl;
	}

	// find halo ids
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
      Node *n = &nodes[i];

      if(n->parentSuper!=NULL && n->parentSuper->value<=linkLength)
        n = n->parentSuper;

			while(n->parent!=NULL && n->parent->value <= linkLength)
			  n = n->parent;

			if(n->nodeId!= nodes[i].nodeId)
				nodes[i].parentSuper = n;

      haloIndex[i] = (n->count >= particleSize) ? n->haloId : -1;
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
/*
	  thrust::device_vector<int>::iterator new_end;

	  tmpIntArray1.resize(numOfParticles);
	  new_end = thrust::remove_copy(haloIndex.begin(), haloIndex.end(), tmpIntArray1.begin(), -1);

	  numOfHaloParticles = new_end - tmpIntArray1.begin();

	  inputX_f.resize(numOfHaloParticles);
    inputY_f.resize(numOfHaloParticles);
    inputZ_f.resize(numOfHaloParticles);
*/
	}

	// clear super parents of all nodes
	void clearSuperParents()
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				clearParentSuper(thrust::raw_pointer_cast(&*nodes.begin())));
	}

	// for a given node, set its super parent to null
	struct clearParentSuper : public thrust::unary_function<int, void>
	{
		Node  *nodes;

		__host__ __device__
		clearParentSuper(Node *nodes) : nodes(nodes) {}

		__host__ __device__
		void operator()(int i)
		{			
      nodes[i].parentSuper = NULL;
		}
	};

	void checkValidMergeTree()
	{
		bool invalid = false;
		for(int i=0; i<numOfParticles; i++)
		{
			Node *n = thrust::raw_pointer_cast(&*nodes.begin());

			int count = 0;
			while(n && n->value <= min_ll)
			{
				count++;
				n = n->parent;
			}

			if(count > 1)
			{
				invalid = true; 
				break;
			}
		}	

		std::cout << std::endl;	

		if(invalid) std::cout << "-- ERROR: invalid merge tree " << std::endl;
		else std::cout << "-- valid merge tree " << std::endl;
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
		cubesInX = (std::ceil((uBoundS.x - lBoundS.x)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.x - lBoundS.x)/cubeLen);
		cubesInY = (std::ceil((uBoundS.y - lBoundS.y)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.y - lBoundS.y)/cubeLen);
		cubesInZ = (std::ceil((uBoundS.z - lBoundS.z)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.z - lBoundS.z)/cubeLen);

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

    std::cout << std::endl;
	  std::cout << "'divideIntoCubes' Time division: " << std::endl << std::flush;
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
    std::cout << std::endl;

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
														cubeLen, lBoundS, cubesInX, cubesInY, cubesInZ));
	}

	// for a given particle, set its cube id
	struct setCubeIdOfParticle : public thrust::unary_function<int, void>
	{
		float  cubeLen;

		Point  lBoundS;
		int    cubesInX, cubesInY, cubesInZ;

		int   *cubeId;
		float *inputX, *inputY, *inputZ;

		__host__ __device__
		setCubeIdOfParticle(float *inputX, float *inputY, float *inputZ,
			int *cubeId, float cubeLen, Point lBoundS,
			int cubesInX, int cubesInY, int cubesInZ) :
			inputX(inputX), inputY(inputY), inputZ(inputZ),
			cubeId(cubeId), cubeLen(cubeLen), lBoundS(lBoundS),
			cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ){}

		__host__ __device__
		void operator()(int i)
		{
			// get x,y,z coordinates for the cube
			int z = (((inputZ[i]-lBoundS.z) / cubeLen)>=cubesInZ) ? cubesInZ-1 : (inputZ[i]-lBoundS.z) / cubeLen;
			int y = (((inputY[i]-lBoundS.y) / cubeLen)>=cubesInY) ? cubesInY-1 : (inputY[i]-lBoundS.y) / cubeLen;
			int x = (((inputX[i]-lBoundS.x) / cubeLen)>=cubesInX) ? cubesInX-1 : (inputX[i]-lBoundS.x) / cubeLen;

			// get cube id
			cubeId[i] = (z*(cubesInX*cubesInY) + y*cubesInX + x);
		}
	};

	// sort particles by cube id
	void sortParticlesByCubeID()
	{
    thrust::stable_sort_by_key(cubeId.begin(), cubeId.end(), particleId.begin());
	}
		
	//*** for each cube, get the size & start of cube particles (in particleId array)
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
	
  //*** for each cube, get its neighbors, here we only set 13 of them not all 26
	void getNeighborsOfCubes()
	{
		neighborsOfCubes.resize(13*numOfCubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				setNeighboringCubes(cubesInX, cubesInY, cubesInZ,	thrust::raw_pointer_cast(&*neighborsOfCubes.begin())));
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
			std::cout << "---- " << ((Edge)edges[i]).srcId << "," << ((Edge)edges[i]).desId << "," << ((Edge)edges[i]).tmp << "," << ((Edge)edges[i]).weight <<  ")" << std::endl;
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
			
			n = ((Node)nodes[i]);
			if(n.parentSuper!=NULL)
			{
				n = *(n.parentSuper);
				std::cout << " - parentSuper (" << n.value << "," << n.nodeId << "," << n.haloId << ")";
			}

			n = ((Node)nodes[i]);
			std::cout << " - sibling ";
			while(n.sibling!=NULL)
			{
				n = *(n.sibling);
				std::cout << "(" << n.value << "," << n.nodeId << "," << n.haloId << ")";
			}

			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << "----------------------" << std::endl << std::endl;
	}



	//------- METHOD : parallel for each cube, get the set of edges & create the submerge tree. Globally combine them

	//---------------- METHOD - Local functions

	// locally, get intra-cube edges for each cube & create the local merge trees
	void localStep()
	{
		tmpNxt.resize(numOfCubes);
		tmpFree.resize(numOfParticles);

		nodes.resize(numOfParticles);
		nodesTmp1.resize(numOfParticles);

    struct timeval begin, mid1, mid2, end, diff1, diff2, diff3;
    gettimeofday(&begin, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				initNodes(thrust::raw_pointer_cast(&*nodes.begin()),
									thrust::raw_pointer_cast(&*nodesTmp1.begin()),
									thrust::raw_pointer_cast(&*tmpFree.begin()),
									numOfParticles));
    gettimeofday(&mid1, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				initNxt(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
								thrust::raw_pointer_cast(&*tmpNxt.begin()),
								thrust::raw_pointer_cast(&*tmpFree.begin())));
    gettimeofday(&mid2, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				createSubMergeTree(thrust::raw_pointer_cast(&*nodes.begin()),
													 thrust::raw_pointer_cast(&*nodesTmp1.begin()),
													 thrust::raw_pointer_cast(&*particleId.begin()),
													 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
												 	 thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
													 thrust::raw_pointer_cast(&*tmpNxt.begin()),
													 thrust::raw_pointer_cast(&*tmpFree.begin()),
													 min_ll));
    gettimeofday(&end, 0);

    std::cout << std::endl;
	  std::cout << "'localStep' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed0: " << seconds1 << " s for initNodes " << std::endl << std::flush;
		timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed1: " << seconds2 << " s for initNxt " << std::endl << std::flush;
		timersub(&end, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed2: " << seconds3 << " s for createSubMergeTree " << std::endl << std::flush;   
	}

	// init the nodes array with node id, halo id, count & set its initial nxt free id
  struct initNodes : public thrust::unary_function<int, void>
  {
		Node *nodes, *nodesTmp1;
		int  *tmpFree;

		int numOfParticles;

		__host__ __device__
		initNodes(Node *nodes, Node *nodesTmp1, int *tmpFree, int numOfParticles) : 
			nodes(nodes), nodesTmp1(nodesTmp1), tmpFree(tmpFree), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {
			nodes[i].nodeId = i;
    	nodes[i].haloId = i;
			nodes[i].count  = 1;

			nodesTmp1[i].nodeId = i+numOfParticles;

			tmpFree[i] = i+1;
    }
 	};

  // finalize the init of tmpFree & tmpNxt arrays
  struct initNxt : public thrust::unary_function<int, void>
  {
    int  *particleSizeOfCubes, *particleStartOfCubes;
		int  *tmpFree, *tmpNxt;

		__host__ __device__
		initNxt(int *particleSizeOfCubes, int *particleStartOfCubes, int *tmpNxt, int *tmpFree) : 
			particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes), tmpNxt(tmpNxt), tmpFree(tmpFree) {}

    __host__ __device__
    void operator()(int i)
    {
			int start = particleStartOfCubes[i];
			int size  = particleSizeOfCubes[i];

		  tmpNxt[i] = (size>0) ? start : -1;

			tmpFree[start+size-1] = -1;
    }
 	};

	struct createSubMergeTree : public thrust::unary_function<int, void>
  {
		Node *nodes, *nodesTmp1;

		int *particleSizeOfCubes, *particleStartOfCubes, *particleId;
		int *tmpNxt, *tmpFree;

	  float min_ll;

		__host__ __device__
		createSubMergeTree(Node *nodes, Node *nodesTmp1, int *particleId,
			int *particleSizeOfCubes, int *particleStartOfCubes, 
			int *tmpNxt, int *tmpFree, float min_ll) :
			nodes(nodes), nodesTmp1(nodesTmp1), particleId(particleId),
			particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes), 
			tmpNxt(tmpNxt), tmpFree(tmpFree), min_ll(min_ll) {}

    __host__ __device__
    void operator()(int i)
    {
			if(particleSizeOfCubes[i]<=1) return;
			
			// get the next free node & set it as the parent
			Node *n = &nodesTmp1[tmpNxt[i]];
			int tmpVal = tmpFree[tmpNxt[i]];
			tmpFree[tmpNxt[i]] = -2;
			tmpNxt[i] = tmpVal;

			int minValue = -1;
			for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
			{
				Node *tmp = &nodes[particleId[j]];

				tmp->parent = n;
				
				if(!n->childS) {	n->childS = tmp;	n->childE = tmp; }
				else {	 n->childE->sibling = tmp;	n->childE = tmp; }

				minValue = (minValue==-1) ? tmp->haloId : (minValue<tmp->haloId ? minValue : tmp->haloId);
			}

			n->value  = min_ll;
      n->haloId = minValue;
      n->count += particleSizeOfCubes[i];
    }
 	};




	//---------------- METHOD - Global functions

  // combine two local merge trees, two cubes at a time
	void globalStep()
	{
		int numOfCubesOri = numOfCubes;
		int sizeP = 2;

		// set new number of cubes
		int numOfCubesOld = numOfCubes;
		numOfCubes = (int)std::ceil(((double)numOfCubes/2));

		std::cout << std::endl;
	
		while(numOfCubes!=numOfCubesOld && numOfCubes>0)
		{
	    struct timeval begin, end, diff;
	    gettimeofday(&begin, 0);
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes, 
				combineFreeLists(thrust::raw_pointer_cast(&*tmpNxt.begin()),
												 thrust::raw_pointer_cast(&*tmpFree.begin()),
												 sizeP, numOfCubesOri));
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				combineMergeTrees(thrust::raw_pointer_cast(&*particleId.begin()),
													thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
													thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
												  thrust::raw_pointer_cast(&*inputX.begin()),
												  thrust::raw_pointer_cast(&*inputY.begin()),
												  thrust::raw_pointer_cast(&*inputZ.begin()),
													thrust::raw_pointer_cast(&*cubeId.begin()),
													thrust::raw_pointer_cast(&*nodesTmp1.begin()),
													thrust::raw_pointer_cast(&*nodes.begin()),
													thrust::raw_pointer_cast(&*tmpNxt.begin()),
													thrust::raw_pointer_cast(&*tmpFree.begin()),
													min_ll, max_ll, sizeP, numOfCubesOri, numOfParticles));
			gettimeofday(&end, 0);

      timersub(&end, &begin, &diff);
      float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
      std::cout << "Time elapsed: " << seconds << " s for numOfCubes " << numOfCubes << std::endl;

			// set new number of cubes & sizeP
			sizeP *= 2;
			numOfCubesOld = numOfCubes;
			numOfCubes = (int)std::ceil(((double)numOfCubes/2));
		}

		#ifdef TEST
			outputMergeTreeDetails("The sub merge trees.."); // output merge tree details
		#endif
	}

	// combine free nodes lists of each cube at this iteration
	struct combineFreeLists : public thrust::unary_function<int, void>
  {
		int   sizeP, numOfCubesOri;

		int  *tmpNxt, *tmpFree;

		__host__ __device__
    combineFreeLists(int *tmpNxt, int *tmpFree, int sizeP, int numOfCubesOri) :
				tmpNxt(tmpNxt), tmpFree(tmpFree), sizeP(sizeP), numOfCubesOri(numOfCubesOri) {}

    __host__ __device__
    void operator()(int i)
    {	
			int cubeStart = sizeP*i;
			int cubeEnd   = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

			int k;
			for(k=cubeStart; k<cubeEnd; k+=sizeP/2)
			{
				if(tmpNxt[k]!=-1) { tmpNxt[cubeStart] = tmpNxt[k];  break; }
			}

			int nxt;
			while(k<cubeEnd)
			{
				nxt = tmpNxt[k];

				while(nxt!=-1 && tmpFree[nxt]!=-1)  nxt = tmpFree[nxt];

				k+=sizeP/2;

				if(k<cubeEnd) tmpFree[nxt] = tmpNxt[k];
			}
		}
	};

	// combine two local merge trees
	struct combineMergeTrees : public thrust::unary_function<int, void>
  {
    float min_ll, max_ll;
		int   sizeP, numOfCubesOri, numOfParticles;

    int  *cubeId;
		int  *tmpNxt, *tmpFree;

    int  *particleId;
    int  *particleStartOfCubes, *particleSizeOfCubes;

		float *inputX, *inputY, *inputZ;

    Node *nodes, *nodesTmp1;

    __host__ __device__
    combineMergeTrees(int *particleId, int *particleSizeOfCubes, int *particleStartOfCubes, 
				float *inputX, float *inputY, float *inputZ,
        int *cubeId, Node *nodesTmp1, Node *nodes,
				int *tmpNxt, int *tmpFree, float min_ll, float max_ll,
				int sizeP, int numOfCubesOri, int numOfParticles) :
        particleId(particleId), particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes), 
				inputX(inputX), inputY(inputY), inputZ(inputZ),
        cubeId(cubeId), nodesTmp1(nodesTmp1), nodes(nodes),
				tmpNxt(tmpNxt), tmpFree(tmpFree),  min_ll(min_ll), max_ll(max_ll),
				sizeP(sizeP), numOfCubesOri(numOfCubesOri), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {	
			int cubeStart = sizeP*i;
			int cubeEnd   = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

			// get the edges
			for(int k=cubeStart; k<cubeStart+sizeP/2; k++)
			{
				for(int l=cubeStart+sizeP/2; l<cubeEnd; l++)
				{

					for(int a=particleStartOfCubes[k]; a<particleStartOfCubes[k]+particleSizeOfCubes[k]; a++)
					{		
						float currentX = inputX[particleId[a]];		float currentY = inputY[particleId[a]];		float currentZ = inputZ[particleId[a]];

						for(int b=particleStartOfCubes[l]; b<particleStartOfCubes[l]+particleSizeOfCubes[l]; b++)
						{
							float otherX = inputX[particleId[b]];		float otherY = inputY[particleId[b]];			float otherZ = inputZ[particleId[b]];

							float xd = (currentX-otherX);  if (xd < 0.0f) xd = -xd;
							float yd = (currentY-otherY);  if (yd < 0.0f) yd = -yd;
							float zd = (currentZ-otherZ);  if (zd < 0.0f) zd = -zd;
																				                                                                                                                    						
							if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
							{
								float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);
								if(dist <= max_ll)
								{
									int srcV = (particleId[a] <= particleId[b]) ? particleId[a] : particleId[b];
									int desV = (srcV == particleId[a]) ? particleId[b] : particleId[a];

									// use this edge, to combine the merge trees
									//-----------------------------------------------------

									Edge e = Edge(srcV, desV, dist);

									Node *src = &nodes[e.srcId];
									Node *des = &nodes[e.desId];

									int weight = (e.weight < min_ll) ? min_ll : e.weight; 

									// find the src & des nodes just below the required weight
									while(src->parent!=NULL && src->parent->value <= weight)	src = src->parent;
									while(des->parent!=NULL && des->parent->value <= weight)	des = des->parent;

									// if src & des already have the same halo id, do NOT do anything
									if(src->haloId==des->haloId) continue;

									int srcCount = src->count;
									int desCount = des->count;

									// get the original parents of src & des nodes
									Node *srcTmp = (src->parent!=NULL) ? src->parent : NULL;
									Node *desTmp = (des->parent!=NULL) ? des->parent : NULL;

									// remove the src & des from the child list of their parents
									if(srcTmp)
									{
										Node *child = srcTmp->childS;
										if(child && child->value==src->value && child->haloId==src->haloId) srcTmp->childS = child->sibling;
										else
										{
											while(child && child->sibling)
											{
												if(child->sibling->value==src->value && child->sibling->haloId==src->haloId)
												{ child->sibling = child->sibling->sibling;	break; }
												child = child->sibling;
											}

											if(child && !child->sibling) srcTmp->childE = child;
										}

										if(!srcTmp->childS) srcTmp->childE = NULL;
									}
									src->parent =NULL;	src->sibling=NULL;

									if(desTmp)
									{
										Node *child = desTmp->childS;
										if(child && child->value==des->value && child->haloId==des->haloId)
											desTmp->childS = child->sibling;
										else
										{
											while(child && child->sibling)
											{
												if(child->sibling->value==des->value && child->sibling->haloId==des->haloId)
												{ child->sibling = child->sibling->sibling;	break; }
												child = child->sibling;
											}

											if(child && !child->sibling) desTmp->childE = child;
										}

										if(!desTmp->childS) desTmp->childE = NULL;
									}
									des->parent =NULL;	des->sibling=NULL;


									// set n node
									Node *n;
									bool freeDes=false;
									if(src->value==weight && des->value==weight) // merge src & des, free des node, set n to src, then connect their children & fix the loop
									{ 
										n = src;					
										Node *child = des->childS;
										while(child!=NULL) { child->parent = n;	child = child->sibling;	}
										if(n->childE)  n->childE->sibling = des->childS;
										else  n->childS = des->childS;
										n->childE = des->childE;	
										freeDes = true;
									}
									else if(src->value==weight) // set des node's parent to be src, set n to src, then fix the loop
									{ 
										n = src;	n->childE->sibling = des;		n->childE = des;		des->parent = n; 	
									}
									else if(des->value==weight) // set src node's parent to be des, set n to des, then fix the loop
									{ 
										n = des;	n->childE->sibling = src;		n->childE = src;		src->parent = n;
									}
									else if(src->value!=weight && des->value!=weight) // create a new node, set this as parent of both src & des, then fix the loop
									{ 
										if(tmpNxt[cubeStart]!=-1)
										{
											n = &nodesTmp1[tmpNxt[cubeStart]];
											int tmpVal = tmpFree[tmpNxt[cubeStart]];
											tmpFree[tmpNxt[cubeStart]] = -2;
											tmpNxt[cubeStart] = tmpVal;

											n->childS = src;	n->childE = des;						
											src->parent = n; 	des->parent = n;
											src->sibling = des;
										}
										else
											std::cout << "***no Free item .... this shouldnt happen*** " << cubeStart << " " << e.weight << " " << min_ll << " " << std::endl;
									}
									n->value  = weight;
									n->count  = src->count + des->count;
									n->haloId = (src->haloId < des->haloId) ? src->haloId : des->haloId;

									if(freeDes && des->nodeId>=numOfParticles)
									{
										// free des node
										int tmpVal = tmpNxt[cubeStart];
										tmpNxt[cubeStart] = des->nodeId-numOfParticles;
										tmpFree[tmpNxt[cubeStart]] = tmpVal;

										des->nodeId = des->nodeId;
										des->haloId = -1;
										des->value  = 0.0f;
										des->count  = 0;
										des->parent = NULL;
										des->parentSuper = NULL;
										des->childS = NULL;
										des->childE = NULL;
										des->sibling = NULL;
									}




									bool done = false;
									while(srcTmp!=NULL && desTmp!=NULL)
									{
										if(srcTmp->value < desTmp->value)
										{
											n->parent = srcTmp;
											if(srcTmp->childE)  srcTmp->childE->sibling = n;
											else  srcTmp->childS = n;
											srcTmp->childE = n;
											srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
											srcCount = srcTmp->count;
											srcTmp->count += desCount;
		
											n = srcTmp;
											srcTmp = srcTmp->parent;

											if(srcTmp)
											{
												Node *child = srcTmp->childS;
												if(child && child->value==n->value && child->haloId==n->haloId)
													srcTmp->childS = child->sibling;
												else
												{
													while(child && child->sibling)
													{
														if(child->sibling->value==n->value && child->sibling->haloId==n->haloId)
														{ child->sibling = child->sibling->sibling;	break; }
														child = child->sibling;
													}

													if(child && !child->sibling) srcTmp->childE = child;
												}

												if(!srcTmp->childS) srcTmp->childE = NULL;
											}
											n->parent  = NULL;	n->sibling = NULL;
										}
										else if(srcTmp->value > desTmp->value)
										{
											n->parent = desTmp;
											if(desTmp->childE)  desTmp->childE->sibling = n;
											else  desTmp->childS = n;
											desTmp->childE = n;
											desTmp->haloId = (desTmp->haloId < n->haloId) ? desTmp->haloId : n->haloId;
											desCount = desTmp->count;
											desTmp->count += srcCount;

											n = desTmp;
											desTmp = desTmp->parent;

											if(desTmp)
											{
												Node *child = desTmp->childS;
												if(child && child->value==n->value && child->haloId==n->haloId)
													desTmp->childS = child->sibling;
												else
												{
													while(child && child->sibling)
													{
														if(child->sibling->value==n->value && child->sibling->haloId==n->haloId)
														{ child->sibling = child->sibling->sibling;	break; }
														child = child->sibling;
													}

													if(child && !child->sibling) desTmp->childE = child;
												}

												if(!desTmp->childS) desTmp->childE = NULL;
											}
											n->parent  = NULL;	n->sibling = NULL;
										}
										else if(srcTmp->value == desTmp->value)
										{
											if(srcTmp->haloId != desTmp->haloId) // combine srcTmp & desTmp			
											{	
												Node *child = desTmp->childS;
												while(child!=NULL) { child->parent = srcTmp;  child = child->sibling; }
												if(srcTmp->childE)  srcTmp->childE->sibling = desTmp->childS;
												else  srcTmp->childS = desTmp->childS;
												srcTmp->childE = desTmp->childE;
												srcTmp->haloId = (srcTmp->haloId < desTmp->haloId) ? srcTmp->haloId : desTmp->haloId;
												srcTmp->count += desTmp->count;
											}

											if(!srcTmp->childS && !srcTmp->childE)
											{
												if(srcTmp->parent)
												{
													Node *child = srcTmp->parent->childS;
													if(child && child->value==srcTmp->value && child->haloId==srcTmp->haloId)
														srcTmp->parent->childS = child->sibling;
													else
													{
														while(child && child->sibling)
														{
															if(child->sibling->value==srcTmp->value && child->sibling->haloId==srcTmp->haloId)
															{ child->sibling = child->sibling->sibling;	break; }
															child = child->sibling;
														}

														if(child && !child->sibling)  srcTmp->parent->childE = child;
													}

													if(!srcTmp->parent->childS) srcTmp->parent->childE=NULL;
												}

												Node *tmp = srcTmp->parent;
												if(srcTmp->nodeId>=numOfParticles)
												{
													int tmpVal = tmpNxt[cubeStart];
													tmpNxt[cubeStart] = srcTmp->nodeId-numOfParticles;
													tmpFree[tmpNxt[cubeStart]] = tmpVal;

													srcTmp->nodeId = srcTmp->nodeId;
													srcTmp->haloId = -1;
													srcTmp->value  = 0.0f;
													srcTmp->count  = 0;
													srcTmp->parent = NULL;
													srcTmp->parentSuper = NULL;
													srcTmp->childS = NULL;
													srcTmp->childE = NULL;
													srcTmp->sibling = NULL;
												}
												srcTmp = tmp;
											}

											if(srcTmp)
											{
												n->parent = srcTmp;
												if(srcTmp->childE)  srcTmp->childE->sibling = n;
												else  srcTmp->childS = n;
												srcTmp->childE = n;
												srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
											}
											else	n->parent =NULL;

											done = true;
											break;
										}
									}



									if(!done && srcTmp!=NULL)
									{
										n->parent = srcTmp;
										if(srcTmp->childE)  srcTmp->childE->sibling = n;
										else  srcTmp->childS = n;
										srcTmp->childE = n;
										srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
										srcCount = srcTmp->count;
										srcTmp->count += desCount;

										n = srcTmp;
										srcTmp = srcTmp->parent;
									}
									while(!done && srcTmp!=NULL)
									{
										srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
										srcCount = srcTmp->count;
										srcTmp->count += desCount;

										n = srcTmp;
										srcTmp = srcTmp->parent;
									}

									if(!done && desTmp!=NULL)
									{		
										n->parent = desTmp;
										if(desTmp->childE)  desTmp->childE->sibling = n;
										else  desTmp->childS = n;
										desTmp->childE = n;
										desTmp->haloId = (desTmp->haloId < n->haloId) ? desTmp->haloId : n->haloId;
										desCount = desTmp->count;
										desTmp->count += srcCount;

										n = desTmp;
										desTmp = desTmp->parent;
									}
									while(!done && desTmp!=NULL)
									{
										desTmp->haloId = (desTmp->haloId < n->haloId) ? desTmp->haloId : n->haloId;
										desCount = desTmp->count;
										desTmp->count += srcCount;

										n = desTmp;
										desTmp = desTmp->parent;
									}

									//-----------------------------------------------------

								}
							}
						}
					}
				}
			}
		}		
	};	
};

}

#endif
