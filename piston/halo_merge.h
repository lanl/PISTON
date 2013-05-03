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
			// globally combine the local merge trees, two cubes at a time

			gettimeofday(&mid2, 0);
			localStep();
			gettimeofday(&mid3, 0);

			neighborsOfCubes.clear();

			std::cout << "-- localStep done" << std::endl;

			gettimeofday(&mid4, 0);
			globalStep();
			gettimeofday(&end, 0);

			clearSuperParents();

			cubeId.clear();
			particleId.clear();
      particleSizeOfCubes.clear();
      particleStartOfCubes.clear();
			neighborsOfCubes.clear();

			edges.clear();
			edgeSizeOfCubes.clear();
			edgeStartOfCubes.clear();

			tmpDoubleArray1.clear();
			tmpIntArray1.clear();
			tmpIntArray2.clear();

			tmpNxt.clear();
			tmpFree.clear();

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

		std::cout << "Number of Particles : " << numOfParticles <<  " Number of Halos found : " << numOfHalos << std::endl << std::endl;
		std::cout << "mergetreeSize : " << mergetreeSize << std::endl << std::endl;
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
	  struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff0, diff1, diff2, diff3, diff4, diff5;
	  gettimeofday(&begin, 0);
		initEdgeArrays(0,0);  		// init arrays needed for storing edges
		gettimeofday(&mid1, 0);
		thrust::fill(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes,0);
		getEdgesPerCube(0,0); 		// for each cube, get the set of edges
		gettimeofday(&mid2, 0);
		sortEdgesPerCube(); 			// for each cube, sort the set of edges by weight
		gettimeofday(&mid3, 0);
		sortCubeIDByParticleID(); // sort cube ids by particle ids
		gettimeofday(&mid4, 0);
		getSubMergeTreePerCube(); // for each cube, get the sub merge tree
		gettimeofday(&mid5, 0);
		edges.clear();
		edgeSizeOfCubes.clear();
		initEdgeArrays(1,13);  		// init arrays needed for storing edges
		thrust::fill(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes,0);
		getEdgesPerCube(1,13); 		// for each cube, get the set of edges
		gettimeofday(&end, 0);

		#ifdef TEST
			outputEdgeDetails("After removing unecessary edges found in each cube.."); // output edge details
			outputMergeTreeDetails("The sub merge trees.."); // output merge tree details
		#endif

    std::cout << std::endl;
    std::cout << "'localStep' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff0);
    float seconds0 = diff0.tv_sec + 1.0E-6*diff0.tv_usec;
    std::cout << "Time elapsed0: " << seconds0 << " s for initEdgeArrays"<< std::endl << std::flush;
    timersub(&mid2, &mid1, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << " s for getEdgesPerCube - Within"<< std::endl << std::flush;
    timersub(&mid3, &mid2, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed3: " << seconds2 << " s for sortEdgesPerCube"<< std::endl << std::flush;
    timersub(&mid4, &mid3, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed4: " << seconds3 << " s for sortCubeIDByParticleID"<< std::endl << std::flush;
    timersub(&mid5, &mid4, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed5: " << seconds4 << " s for getSubMergeTreePerCube"<< std::endl << std::flush;
    timersub(&end, &mid5, &diff5);
    float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
    std::cout << "Time elapsed6: " << seconds5 << " s for getEdgesPerCube - Across"<< std::endl << std::flush;
    std::cout << std::endl;

	}

	// for each cube, init arrays needed for storing edges
	void initEdgeArrays(int start, int end)
	{
		// for each cube, set the space required for storing edges
		edgeSizeOfCubes.resize(numOfCubes);

		for(int i=start; i<=end; i++)
		{
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
					getEdgesSizes(thrust::raw_pointer_cast(&*neighborsOfCubes.begin()),
									      thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
									      thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
									      thrust::raw_pointer_cast(&*inputX.begin()),
									      thrust::raw_pointer_cast(&*inputY.begin()),
									      thrust::raw_pointer_cast(&*inputZ.begin()),
									      thrust::raw_pointer_cast(&*particleId.begin()),
									      max_ll, i,
									      thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin())));
		}
		edgeStartOfCubes.resize(numOfCubes);
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes, edgeStartOfCubes.begin());

		numOfEdges = edgeStartOfCubes[numOfCubes-1] + edgeSizeOfCubes[numOfCubes-1];

		std::cout << numOfEdges << std::endl;

		// init edge arrays
		edges.resize(numOfEdges);
	}

  // for each cube, get the size of edges for allocations of edge array
  struct getEdgesSizes : public thrust::unary_function<int, void>
  {
    int    num;
    float  max_ll;
    float *inputX, *inputY, *inputZ;

    int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;
    int   *neighborsOfCubes;

    int  *edgeSizeOfCubes;

    __host__ __device__
    getEdgesSizes(int *neighborsOfCubes, int *particleStartOfCubes, int *particleSizeOfCubes,
        float *inputX, float *inputY, float *inputZ,
        int *particleId, float max_ll, int num, int *edgeSizeOfCubes) :
        neighborsOfCubes(neighborsOfCubes),
        particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
        inputX(inputX), inputY(inputY), inputZ(inputZ),
        particleId(particleId), max_ll(max_ll), num(num), edgeSizeOfCubes(edgeSizeOfCubes) {}

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

            if(xd<=max_ll && yd<=max_ll && zd<=max_ll && std::sqrt(xd*xd + yd*yd + zd*zd) <= max_ll)
              size++;
         }
       }

       edgeSizeOfCubes[i] += size;
    }
  };

	// for each cube, get the set of edges
	void getEdgesPerCube(int start, int end)
	{	
		for(int i=start; i<=end; i++)
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
		
		Edge  *edges;
		int   *edgeStartOfCubes, *edgeSizeOfCubes;

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
			 int tmpStart = -1;
			 int cube = (num==0) ? i : neighborsOfCubes[13*i+(num-1)];

			 if(cube==-1 || particleSizeOfCubes[i]==0 || particleSizeOfCubes[cube]==0) return;

			 // for each particle in cube
			 for(int j=particleStartOfCubes[cube]; j<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; j++)
			 {
					bool edgeFound = false;

					float currentX = inputX[particleId[j]];
				  float currentY = inputY[particleId[j]];
				  float currentZ = inputZ[particleId[j]];

				  // compare with particles in this cube
				  int start = (num==0) ? j+1 : particleStartOfCubes[i];
				  for(int k=start; k<particleStartOfCubes[i]+particleSizeOfCubes[i]; k++)
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
								edgeFound = true;

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

  // for each cube, sort the edges by weight
  void sortEdgesPerCube()
  {
    tmpDoubleArray1.resize(numOfEdges); // stores the keys

    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
        setValuesAndKeys(thrust::raw_pointer_cast(&*edges.begin()),
                 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
                 max_ll,
                 thrust::raw_pointer_cast(&*tmpDoubleArray1.begin())));

    thrust::stable_sort_by_key(tmpDoubleArray1.begin(), tmpDoubleArray1.end(), edges.begin());

    tmpDoubleArray1.clear();
  }

  // set values & keys arrays needed for edge sorting
  struct setValuesAndKeys : public thrust::unary_function<int, void>
  {
    float max_ll;

    double *keys;

    Edge *edges;
    int  *edgeSizeOfCubes, *edgeStartOfCubes;

    __host__ __device__
    setValuesAndKeys(Edge *edges, int *edgeSizeOfCubes,int *edgeStartOfCubes,
        float max_ll, double *keys) :
        edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
        max_ll(max_ll), keys(keys) {}

    __host__ __device__
    void operator()(int i)
    {
      for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
        keys[j] = (double)(edges[j].weight + (double)2*i*max_ll); 
    }
  };

  // sort cube id by particle id
  void sortCubeIDByParticleID()
  {
    tmpIntArray1.resize(numOfParticles);

    thrust::copy(particleId.begin(), particleId.end(), tmpIntArray1.begin());
    thrust::stable_sort_by_key(tmpIntArray1.begin(), tmpIntArray1.end(), cubeId.begin());

    tmpIntArray1.clear();
  }

  // for each cube, compute the sub merge trees
	void getSubMergeTreePerCube()
	{
		tmpNxt.resize(numOfCubes);
		tmpFree.resize(numOfParticles);

		nodes.resize(numOfParticles);
		nodesTmp1.resize(numOfParticles);

	  struct timeval begin, mid1, end, diff0, diff1;
	  gettimeofday(&begin, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				initNodes(thrust::raw_pointer_cast(&*nodes.begin()),
									thrust::raw_pointer_cast(&*nodesTmp1.begin()),
									thrust::raw_pointer_cast(&*tmpFree.begin()),
									numOfParticles));
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				initNxt(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
								thrust::raw_pointer_cast(&*tmpNxt.begin()),
								thrust::raw_pointer_cast(&*tmpFree.begin())));
	  gettimeofday(&mid1, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				getSubMergeTree(thrust::raw_pointer_cast(&*edges.begin()),
				                thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
												thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
												thrust::raw_pointer_cast(&*particleId.begin()),
												thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
												thrust::raw_pointer_cast(&*cubeId.begin()),
												thrust::raw_pointer_cast(&*tmpNxt.begin()),
												thrust::raw_pointer_cast(&*tmpFree.begin()),
												thrust::raw_pointer_cast(&*nodesTmp1.begin()),
												thrust::raw_pointer_cast(&*nodes.begin()),
												min_ll));
	  gettimeofday(&end, 0);

/*
    std::cout << std::endl;
    std::cout << "'getSubMergeTreePerCube' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff0);
    float seconds0 = diff0.tv_sec + 1.0E-6*diff0.tv_usec;
    std::cout << "Time elapsed0: " << seconds0 << " s for "<< std::endl << std::flush;
    timersub(&end, &mid1, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << " s for "<< std::endl << std::flush;
    std::cout << std::endl;
*/
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

			if(size>0) tmpNxt[i] = start;
			else tmpNxt[i] = -1;

			tmpFree[start+size-1] = -1;
    }
 	};

  // for a given cube, compute its sub merge tree
  struct getSubMergeTree : public thrust::unary_function<int, void>
  {
    float min_ll;

    Edge *edges;
    int  *edgeStartOfCubes, *edgeSizeOfCubes;

    int  *particleId;
    int  *particleStartOfCubes;
    int  *cubeId;

		int  *tmpNxt, *tmpFree;

    Node *nodes, *nodesTmp1;

    __host__ __device__
    getSubMergeTree(Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes,
        int *particleId, int *particleStartOfCubes,
        int *cubeId, int *tmpNxt, int *tmpFree, Node *nodesTmp1, Node *nodes, float min_ll) :
        edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
        particleId(particleId), particleStartOfCubes(particleStartOfCubes),
        cubeId(cubeId), tmpNxt(tmpNxt), tmpFree(tmpFree), nodesTmp1(nodesTmp1), nodes(nodes), min_ll(min_ll) {}

    __host__ __device__
    void operator()(int i)
    {
      for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
      {
        Edge e = ((Edge)edges[j]);

        // get the two merge tree leaf nodes to compare
        Node *tmp1, *tmp2, *tmp3, *tmp4;

        tmp1 = &nodes[e.srcId];
        tmp2 = &nodes[e.desId];

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

						if(n1->childE) { n1->childE->sibling = n2;  n1->childE = n2; }
						else { n1->childS = n2;   n1->childE = n2; }

            if(e.weight < min_ll)
						{
							if(n2->childS!=NULL)
							{
								n1->childE->sibling = n2->childS;
								n1->childE = n2->childE;
		
								Node *tmp = n2->childS;
								tmp->parent = n1;
								tmp->parentSuper = n1;
							
								while(tmp->sibling!=NULL) { tmp = tmp->sibling;		tmp->parent = n1;		tmp->parentSuper = n1; }
							}
						}
          }
          else
          {
						Node *n = &nodesTmp1[tmpNxt[i]];
						int tmpVal = tmpFree[tmpNxt[i]];
						tmpFree[tmpNxt[i]] = -2;
						tmpNxt[i] = tmpVal;

            n->value  = e.weight;
            n->haloId = minValue;
            n->count += (tmp3->count + tmp4->count);

            tmp3->parent      = n;
            tmp4->parent      = n;
            tmp1->parentSuper = n;
            tmp2->parentSuper = n;

						n->childS = tmp3;
						n->childE = tmp4;
						tmp3->sibling = tmp4;

            if(e.weight < min_ll)
						{						
							if(tmp3->childS!=NULL)
							{
								n->childS = tmp3->childS;
		
								Node *tmp = tmp3->childS;
								tmp->parent = n;
								tmp->parentSuper = n;
							
								while(tmp->sibling!=NULL) { tmp = tmp->sibling;		tmp->parent = n;		tmp->parentSuper = n;	}
							}
						
							if(tmp4->childS!=NULL)
							{
								n->childE = tmp4->childE;

								Node *tmp = tmp4->childS;
								tmp->parent = n;
								tmp->parentSuper = n;
							
								while(tmp->sibling!=NULL) { tmp = tmp->sibling;		tmp->parent = n;		tmp->parentSuper = n; }
							}
						
							if(tmp3->childE!=NULL)  tmp3->childE->sibling = (tmp4->childS!=NULL) ? tmp4->childS : tmp4;
							else tmp3->sibling = (tmp4->childS!=NULL) ? tmp4->childS : tmp4;
						}
          }
        }
      }
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

		if(numOfEdges==0) return;

//		std::cout << "edgeSizeOfCubes	 "; thrust::copy(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubesOri, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//		std::cout << "particleSizeOfCubes	 "; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubesOri, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
	
		while(numOfCubes!=numOfCubesOld && numOfCubes>0)
		{
	    struct timeval begin, end, diff;
	    gettimeofday(&begin, 0);
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes, 
				combineFreeLists(thrust::raw_pointer_cast(&*tmpNxt.begin()),
												 thrust::raw_pointer_cast(&*tmpFree.begin()),
												 sizeP, numOfCubesOri));
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				combineMergeTrees(thrust::raw_pointer_cast(&*edges.begin()),
						              thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
													thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
													thrust::raw_pointer_cast(&*particleId.begin()),
													thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
													thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
													thrust::raw_pointer_cast(&*cubeId.begin()),
													thrust::raw_pointer_cast(&*nodesTmp1.begin()),
													thrust::raw_pointer_cast(&*nodes.begin()),
													thrust::raw_pointer_cast(&*tmpNxt.begin()),
													thrust::raw_pointer_cast(&*tmpFree.begin()),
													min_ll, sizeP, numOfCubesOri, numOfParticles));
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
        removeExtraNodes(thrust::raw_pointer_cast(&*nodes.begin()), min_ll));
			gettimeofday(&end, 0);

      timersub(&end, &begin, &diff);
      float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
      std::cout << "Time elapsed: " << seconds << " s for numOfCubes " << numOfCubes << std::endl << std::flush;

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
    float min_ll;
		int   sizeP, numOfCubesOri, numOfParticles;

		int  *tmpNxt, *tmpFree;

    Edge *edges;
    int  *edgeStartOfCubes, *edgeSizeOfCubes;

    int  *particleId;
    int  *particleStartOfCubes, *particleSizeOfCubes;
    int  *cubeId;

    Node *nodes, *nodesTmp1;

    __host__ __device__
    combineMergeTrees(Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes,
        int *particleId, int *particleSizeOfCubes, int *particleStartOfCubes, 
        int *cubeId, Node *nodesTmp1, Node *nodes,
				int *tmpNxt, int *tmpFree, float min_ll,
				int sizeP, int numOfCubesOri, int numOfParticles) :
        edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
        particleId(particleId), particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes), 
        cubeId(cubeId), nodesTmp1(nodesTmp1), nodes(nodes),
				tmpNxt(tmpNxt), tmpFree(tmpFree),  min_ll(min_ll),
				sizeP(sizeP), numOfCubesOri(numOfCubesOri), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {	
			int cubeStart = sizeP*i;
			int cubeEnd   = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

			bool need1 = false, need2 = false;
			for(int k=cubeStart; k<cubeStart+sizeP/2; k++)
			{
				if(particleSizeOfCubes[k]!=0)
				{
					need1 = true;
					break;
				}
			}
			for(int k=cubeStart+sizeP/2; k<cubeEnd; k++)
			{
				if(particleSizeOfCubes[k]!=0)
				{
					need2 = true;
					break;
				}
			}

			if(!(need1 && need2)) return;

			for(int k=cubeStart; k<cubeEnd; k++)
			{
				int size = 0;
				for(int j=edgeStartOfCubes[k]; j<edgeStartOfCubes[k]+edgeSizeOfCubes[k]; j++)
		    {
		      Edge e = ((Edge)edges[j]);

					if( !(cubeId[e.srcId]>=cubeStart && cubeId[e.srcId]<cubeEnd) || !(cubeId[e.desId]>=cubeStart && cubeId[e.desId]<cubeEnd) )
					{
						edges[edgeStartOfCubes[k] + size++] = e;
						continue;
					}

					Node *src = &nodes[e.srcId];
					Node *des = &nodes[e.desId];

					// find the nodes just below the required weight
				  while(src->parent!=NULL && src->parent->value <= e.weight)	src = src->parent;
				  while(des->parent!=NULL && des->parent->value <= e.weight)	des = des->parent;

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
            if(child && child->value==src->value && child->haloId==src->haloId)
              srcTmp->childS = child->sibling;
            else
            {
              while(child && child->sibling)
              {
                if(child->sibling->value==src->value && child->sibling->haloId==src->haloId)
                {
                  child->sibling = child->sibling->sibling;
                  break;
                }
                child = child->sibling;
              }

              if(child && !child->sibling) srcTmp->childE = child;
            }

						if(!srcTmp->childS) srcTmp->childE=NULL;
          }
          src->parent =NULL;
          src->sibling=NULL;

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
                {
                  child->sibling = child->sibling->sibling;
                  break;
                }
                child = child->sibling;
              }

              if(child && !child->sibling) desTmp->childE = child;
            }

						if(!desTmp->childS) desTmp->childE=NULL;
          }
          des->parent =NULL;
          des->sibling=NULL;


					Node *n;
					if(src->value==e.weight && des->value==e.weight) // merge src & des, free des node, set n to des, then connect their children & fix the loop
					{ 
						n = src;					
						Node *child = des->childS;
						while(child!=NULL) { child->parent = n;	child = child->sibling;	}
						n->childE->sibling = des->childS;
						n->childE = des->childE;

						// free des node
						int tmpVal = tmpNxt[cubeStart];
						tmpNxt[cubeStart] = des->nodeId-numOfParticles;
						tmpFree[tmpNxt[cubeStart]] = tmpVal;

						des->nodeId = srcTmp->nodeId;
						des->haloId = -1;
						des->value  = 0.0f;
						des->count  = 0;
            des->parent = NULL;
            des->parentSuper = NULL;
            des->childS = NULL;
            des->childE = NULL;
            des->sibling = NULL;
					}
					else if(src->value==e.weight) // set des node's parent to be src, set n to src, then fix the loop
					{ 
						n = src;
						n->childE->sibling = des;
						n->childE = des;
						des->parent = n; 	
					}
					else if(des->value==e.weight) // set src node's parent to be des, set n to des, then fix the loop
					{ 
						n = des;
						n->childE->sibling = src;
						n->childE = src;
						src->parent = n;
					}
					else if(src->value!=e.weight && des->value!=e.weight) // create a new node, set this as parent of both src & des, then fix the loop
					{ 
						if(tmpNxt[cubeStart]!=-1)
						{
	            n = &nodesTmp1[tmpNxt[cubeStart]];
	            int tmpVal = tmpFree[tmpNxt[cubeStart]];
	            tmpFree[tmpNxt[cubeStart]] = -2;
	            tmpNxt[cubeStart] = tmpVal;
						}
						else
							std::cout << "***no Free item .... this shouldnt happen***" << cubeStart << std::endl;

						n->childS = src;	n->childE = des;						
						src->parent = n; 	des->parent = n;
						src->sibling = des;
					}
					n->value  = e.weight;
					n->count  = src->count + des->count;
					n->haloId = (src->haloId < des->haloId) ? src->haloId : des->haloId;


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
                    {
                      child->sibling = child->sibling->sibling;
                      break;
                    }
                    child = child->sibling;
                  }

                  if(child && !child->sibling) srcTmp->childE = child;
                }

								if(!srcTmp->childS) srcTmp->childE=NULL;
              }
              n->parent =NULL;
              n->sibling=NULL;
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
                    {
                      child->sibling = child->sibling->sibling;
                      break;
                    }
                    child = child->sibling;
                  }

                  if(child && !child->sibling) desTmp->childE = child;
                }

								if(!desTmp->childS) desTmp->childE=NULL;
              }
              n->parent =NULL;
              n->sibling=NULL;
						}
						else if(srcTmp->value == desTmp->value)
						{
							if(srcTmp->haloId != desTmp->haloId) // combine srcTmp & desTmp			
							{	
								Node *child = desTmp->childS;
								while(child!=NULL) { child->parent = srcTmp;  child = child->sibling; }
								srcTmp->childE->sibling = desTmp->childS;
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
		                  {
		                    child->sibling = child->sibling->sibling;
		                    break;
		                  }
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
							else
								n->parent =NULL;

							done = true;
							break;
						}
					}



					if(!done && srcTmp!=NULL)
					{
						n->parent = srcTmp;
						srcTmp->childE->sibling = n;
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
	          desTmp->childE->sibling = n;
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
				}

				edgeSizeOfCubes[k] = size;
			}
		}		
	};

	// remove extra nodes less than the min_ll
	struct removeExtraNodes : public thrust::unary_function<int, void>
	{
	  Node *nodes;

	  float min_ll;

	  __host__ __device__
	  removeExtraNodes(Node *nodes, float min_ll) : nodes(nodes), min_ll(min_ll) {}

	  __host__ __device__
    void operator()(int i)
    {
	    Node *n = &nodes[i];

			if(n->value == min_ll) return;

      while(n->parent!=NULL && (n->parent)->value <= min_ll) n = n->parent;

      if(n->nodeId != nodes[i].nodeId)  nodes[i].parent = n;
    }
	};
};

}

#endif
