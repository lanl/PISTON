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
		int   nodeId; // particle Id
    int   haloId; // halo Id, store the min particle id of the particles in this halo

		int   count;  // store number of particles in the halo at this level
		float value;  // function value of this node (which is the distance value)

		Node *parent, *parentSuper; 
		Node *childE, *childS, *sibling;       

		__host__ __device__
		Node() { nodeId=-1; value=0.0f; haloId=-1; count=0;	
						 parent=NULL; parentSuper=NULL; 
						 childS=NULL; childE=NULL; sibling=NULL; }

		__host__ __device__
		Node(int nodeId, float value, int haloId, int count, Node *parent, Node *parentSuper) :
			nodeId(nodeId), value(value), haloId(haloId), count(count), parent(parent), parentSuper(parentSuper) {}
	};

	struct Edge
	{
		int srcId, desId; // src & des Ids of particles
		float weight;			// weight of the edge (which is the distance value)

		__host__ __device__
		Edge() { srcId=-1; desId=-1; weight=-1; }
		
		__host__ __device__
		Edge(int srcId, int desId, float weight) : srcId(srcId), desId(desId), weight(weight) {}
	};

	float cubeLen;					// length of the cube
	float max_ll, min_ll;   // maximum & minimum linking lengths
	
	float totalTime; 				// total time taken for halo finding

	int mergetreeSize;      // total size of the global merge tree
  int numOfEdges;         // total number of edges in space

	int  side, size, ite; 	// variable needed to determine the neighborhood cubes

	bool ignoreEmptyCubes;						// flag to whether or not to ignore empty cubes in local step
	int numOfCubes;					   	 			// total number of cubes in space
	int  cubesNonEmpty, cubesEmpty;		// total number of nonempty & empty cubes in space
	int cubesInX, cubesInY, cubesInZ; // number of cubes in each dimension
	
  thrust::device_vector<int>   particleId; 						// for each particle, particle id
	thrust::device_vector<int>   particleSizeOfCubes; 	// number of particles in cubes
	thrust::device_vector<int>   particleStartOfCubes;	// stratInd of cubes  (within particleId)

  thrust::device_vector<int>   cubeId; // for each particle, cube id
  thrust::device_vector<int>   cubeMapping, cubeMappingInv; // mapping which seperates empty & nonempty cubes
  thrust::device_vector<int>   sizeOfChunks;  // size of each chunk of cubes
  thrust::device_vector<int>   startOfChunks; // start of each chunk of cubes

	thrust::device_vector<Node>  nodes; 		// leaf nodes of merge tree
  thrust::device_vector<Node>  nodesTmp1; // parent nodes of merge tree

	thrust::device_vector<Edge>  edges;						 // edge of cubes
	thrust::device_vector<int>   edgeSizeOfCubes;  // size of edges in cubes
	thrust::device_vector<int>   edgeStartOfCubes; // start of edges in cubes
	
	thrust::device_vector<int>   tmpIntArray;		   // temperary array 
	thrust::device_vector<int>   tmpNxt, tmpFree;  // stores details of free items in merge tree

	halo_merge(float min_linkLength, float max_linkLength, bool ignore, std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1) : halo(filename, format, n, np, rL)
	{
		if(numOfParticles!=0)
		{
			struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3;

			//---- init stuff

			min_ll  = min_linkLength; // get min_linkinglength
			max_ll  = max_linkLength; // get max_linkinglength
			cubeLen = min_ll / std::sqrt(3); // min_ll*min_ll = 3*cubeLen*cubeLen

			ignoreEmptyCubes = ignore;

			if(cubeLen <= 0) { std::cout << "--ERROR : plase specify a valid cubeLen... current cubeLen is " << cubeLen << std::endl; return; }

			initDetails();

			std::cout << "lBoundS " << lBoundS.x << " " << lBoundS.y << " " << lBoundS.z << std::endl;
			std::cout << "uBoundS " << uBoundS.x << " " << uBoundS.y << " " << uBoundS.z << std::endl;

			//---- divide space into cubes
			gettimeofday(&begin, 0);
			divideIntoCubes();
			gettimeofday(&mid1, 0);

			std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ") ... cubeLen " << cubeLen << std::endl;

			#ifdef TEST
				outputCubeDetails("init cube details"); // output cube details
			#endif

			//------- METHOD :
	    // parallel for each cube, create the local merge tree & get the set of edges.
			// globally combine the cubes, two cubes at a time by considering the edges

			gettimeofday(&mid2, 0);
			localStep();
			gettimeofday(&mid3, 0);

			std::cout << "-- localStep done" << std::endl;

			gettimeofday(&mid4, 0);
			globalStep();
			gettimeofday(&end, 0);

			std::cout << "-- globalStep done" << std::endl;

			//checkValidMergeTree(); 
			//getSizeOfMergeTree(); 
			clearSuperParents();

			particleId.clear();
      particleSizeOfCubes.clear();
      particleStartOfCubes.clear();

			cubeId.clear();
			cubeMapping.clear();
			cubeMappingInv.clear();
			sizeOfChunks.clear();
			startOfChunks.clear();

			edges.clear();
			edgeSizeOfCubes.clear();
			edgeStartOfCubes.clear();

			tmpIntArray.clear();

			tmpNxt.clear();
			tmpFree.clear();

			std::cout << std::endl;
			timersub(&mid1, &begin, &diff1);
			float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
			std::cout << "Time elapsed: " << seconds1 << " s for dividing space into cubes"<< std::endl << std::flush;
			timersub(&mid3, &mid2, &diff2);
			float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
			std::cout << "Time elapsed: " << seconds2 << " s for localStep - finding inter-cube edges"<< std::endl << std::flush;
			timersub(&end, &mid4, &diff3);
			float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
			std::cout << "Time elapsed: " << seconds3 << " s for globalStep - adjusting the merge trees"<< std::endl << std::flush;
			totalTime = seconds1 + seconds2 + seconds3;
			std::cout << "Total time elapsed: " << totalTime << " s for constructing the global merge tree" << std::endl << std::endl;
			
		}
	}

	void operator()(float linkLength, int  particleSize)
	{
		clear();

		linkLength   = linkLength;
		particleSize = particleSize;

		// if no valid particles, return
		if(numOfParticles==0) return;

		struct timeval begin, end, diff;

		gettimeofday(&begin, 0);
		findHalos(linkLength, particleSize); // find halos
		gettimeofday(&end, 0);

		timersub(&end, &begin, &diff);
		float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		totalTime +=seconds;

		std::cout << "Total time elapsed: " << seconds << " s for finding halos at linking length " << linkLength << " and has particle size >= " << particleSize << std::endl << std::endl;

		getNumOfHalos();      // get the unique halo ids & set numOfHalos
		getHaloParticles();   // get the halo particles & set numOfHaloParticles
		setColors();          // set colors to halos
		writeHaloResults();	  // write halo results

		std::cout << "Number of Particles   : " << numOfParticles << std::endl;
		std::cout << "Number of Halos found : " << numOfHalos << std::endl;
		std::cout << "Merge tree size : " << mergetreeSize << std::endl;
    std::cout << "Min_ll  : " << min_ll  << std::endl;
    std::cout << "Max_ll  : " << max_ll << std::endl << std::endl;
		std::cout << "-----------------------------" << std::endl << std::endl;
	}

	// find halo ids 
	void findHalos(float linkLength, int particleSize)
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setHaloId(thrust::raw_pointer_cast(&*nodes.begin()),
						 		  thrust::raw_pointer_cast(&*haloIndex.begin()),
						      linkLength, particleSize));
	}

	// for a given node set its halo id, for particles in filtered halos set id to -1
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
	void getSizeOfMergeTree()
	{
		thrust::host_vector<Node>  nodes_h;
		nodes_h.resize(numOfParticles);

		thrust::copy(nodes.begin(), nodes.end(), nodes_h.begin());

		mergetreeSize = 0;
		for(int i=0; i<numOfParticles; i++)
		{
			Node *n = &nodes_h[i];

			while(n->parent!=NULL)
			{
				n = (n->parent);
				if(n->nodeId != -2) mergetreeSize++;
				n->nodeId = -2;								
			}
		}

		mergetreeSize += numOfParticles;
	}

	// get particles of valid halos & get number of halo particles 
	void getHaloParticles()
	{
	  thrust::device_vector<int>::iterator new_end;

	  tmpIntArray.resize(numOfParticles);
		thrust::sequence(tmpIntArray.begin(), tmpIntArray.begin()+numOfParticles);

	  new_end = thrust::remove_if(tmpIntArray.begin(), tmpIntArray.begin()+numOfParticles,
				invalidHalo(thrust::raw_pointer_cast(&*haloIndex.begin())));

	  numOfHaloParticles = new_end - tmpIntArray.begin();

		haloIndex_f.resize(numOfHaloParticles);
	  inputX_f.resize(numOfHaloParticles);
    inputY_f.resize(numOfHaloParticles);
    inputZ_f.resize(numOfHaloParticles);

		thrust::gather(tmpIntArray.begin(), tmpIntArray.begin()+numOfHaloParticles, haloIndex.begin(), haloIndex_f.begin());
		thrust::gather(tmpIntArray.begin(), tmpIntArray.begin()+numOfHaloParticles, inputX.begin(), inputX_f.begin());
		thrust::gather(tmpIntArray.begin(), tmpIntArray.begin()+numOfHaloParticles, inputY.begin(), inputY_f.begin());
		thrust::gather(tmpIntArray.begin(), tmpIntArray.begin()+numOfHaloParticles, inputZ.begin(), inputZ_f.begin());
	}

	// given a haloIndex of a particle, check whether this particle DOES NOT belong to a halo
	struct invalidHalo : public thrust::unary_function<int, bool>
	{
		int  *haloIndex;

		__host__ __device__
		invalidHalo(int *haloIndex) : haloIndex(haloIndex) {}

		__host__ __device__
		bool operator()(int i)
		{			
      return (haloIndex[i]==-1);
		}
	};

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

	// check whether the merge tree is valid or not
	void checkValidMergeTree()
	{
		thrust::host_vector<Node>  nodes_h;
		nodes_h.resize(numOfParticles);

		thrust::copy(nodes.begin(), nodes.end(), nodes_h.begin());
	
		bool invalid = false;
		for(int i=0; i<numOfParticles; i++)
		{
			Node *n = &nodes_h[i];

			int count = 0;
			while(n && n->value <= min_ll)
			{ n = n->parent;	count++; }

			// if a node has more than one parent node with its value <= min_ll, the tree is invalid
			if(count > 2)
			{ invalid = true; 	std::cout << i << " " << count << std::endl; break; }
		}	
		std::cout << std::endl;	

		if(invalid) std::cout << "-- ERROR: invalid merge tree " << std::endl;
		else std::cout << "-- valid merge tree " << std::endl;
	}




	//------- init stuff
	void initDetails()
	{
		initParticleIds();	// set particle ids
		setNumberOfCubes();	// get total number of cubes
	}

  // set initial particle ids
	void initParticleIds()
	{
		particleId.resize(numOfParticles);
		thrust::sequence(particleId.begin(), particleId.end());
	}

	// get total number of cubes
	void setNumberOfCubes()
	{
		cubesInX = (std::ceil((uBoundS.x - lBoundS.x)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.x - lBoundS.x)/cubeLen);
		cubesInY = (std::ceil((uBoundS.y - lBoundS.y)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.y - lBoundS.y)/cubeLen);
		cubesInZ = (std::ceil((uBoundS.z - lBoundS.z)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.z - lBoundS.z)/cubeLen);

		numOfCubes = cubesInX*cubesInY*cubesInZ; // set number of cubes
	}



	//------- divide space into cubes
	void divideIntoCubes()
	{
	  struct timeval begin, mid1, mid2, end, diff0, diff1, diff2;
	  gettimeofday(&begin, 0);
		setCubeIds();		      		// for each particle, set cube id
	  gettimeofday(&mid1, 0);
		sortParticlesByCubeID();  // sort Particles by cube Id
	  gettimeofday(&mid2, 0);
		getSizeAndStartOfCubes(); // for each cube, count its particles
	  gettimeofday(&end, 0);
    std::cout << std::endl;
	  std::cout << "'divideIntoCubes' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff0);
    float seconds0 = diff0.tv_sec + 1.0E-6*diff0.tv_usec;
    std::cout << "Time elapsed0: " << seconds0 << " s for setCubeIds"<< std::endl << std::flush;
    timersub(&mid2, &mid1, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << " s for sortParticlesByCubeID"<< std::endl << std::flush;
    timersub(&end, &mid2, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed2: " << seconds2 << " s for getSizeAndStartOfCubes"<< std::endl << std::flush;
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
			
			cubeId[i] = (z*(cubesInX*cubesInY) + y*cubesInX + x); // get cube id
		}
	};

	// sort particles by cube id
	void sortParticlesByCubeID()
	{
    thrust::stable_sort_by_key(cubeId.begin(), cubeId.end(), particleId.begin());
	}
		
	// sort cube id by particles
	void sortCubeIDByParticles()
	{
    thrust::stable_sort_by_key(particleId.begin(), particleId.end(), cubeId.begin());
	}

//------------ TODO : this method DOES NOT scale very well. try to change it 
	// for each cube, get the size & start of cube particles (in particleId array)
	void getSizeAndStartOfCubes()
	{
		cubeMapping.resize(numOfCubes);
		cubeMappingInv.resize(numOfCubes);

		particleSizeOfCubes.resize(numOfCubes);
		particleStartOfCubes.resize(numOfCubes);

		tmpIntArray.resize(numOfCubes);

		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;

		new_end = thrust::reduce_by_key(cubeId.begin(), cubeId.end(), ConstantIterator(1), cubeMapping.begin(), tmpIntArray.begin());

		cubesNonEmpty = thrust::get<0>(new_end) - cubeMapping.begin();
		cubesEmpty    = numOfCubes - cubesNonEmpty;	

		if(ignoreEmptyCubes) // get the size & start details for only non empty cubes
		{
			thrust::copy(tmpIntArray.begin(), tmpIntArray.begin()+cubesNonEmpty, particleSizeOfCubes.begin());

			thrust::set_difference(CountingIterator(0), CountingIterator(0)+numOfCubes, cubeMapping.begin(), cubeMapping.begin()+cubesNonEmpty, cubeMapping.begin()+cubesNonEmpty);

			thrust::scatter(CountingIterator(0), CountingIterator(0)+numOfCubes, cubeMapping.begin(), cubeMappingInv.begin());
		}
		else // get the size & start details for all cubes
		{
			thrust::scatter(tmpIntArray.begin(), tmpIntArray.begin()+cubesNonEmpty, cubeMapping.begin(), particleSizeOfCubes.begin());

			thrust::sequence(cubeMapping.begin(), cubeMapping.end());
			thrust::sequence(cubeMappingInv.begin(), cubeMappingInv.end());
		}

		thrust::exclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, particleStartOfCubes.begin());

		tmpIntArray.clear();
	}
	


	//------- output results

	// print cube details from device vectors
	void outputCubeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;
		std::cout << "inputX		"; thrust::copy(inputX.begin(), inputX.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputY		"; thrust::copy(inputY.begin(), inputY.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputZ		"; thrust::copy(inputZ.begin(), inputZ.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << std::endl << "-- Outputs------------" << std::endl << std::endl;

		std::cout << "-- Dim    (" << lBoundS.x << "," << lBoundS.y << "," << lBoundS.z << "), (";
		std::cout << uBoundS.x << "," << uBoundS.y << "," << uBoundS.z << ")" << std::endl;
		std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;

		std::cout << std::endl << "----------------------" << std::endl << std::endl;

		std::cout << "particleId 	"; thrust::copy(particleId.begin(), particleId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "cubeID		"; thrust::copy(cubeId.begin(), cubeId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputX		"; thrust::copy(inputX.begin(), inputX.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputY		"; thrust::copy(inputY.begin(), inputY.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputZ		"; thrust::copy(inputZ.begin(), inputZ.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "startOfCube	"; thrust::copy(particleStartOfCubes.begin(), particleStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
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

		for(int i=0; i<numOfCubes; i++)
		{
			for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)	
			{
				std::cout << "---- " << ((Edge)edges[j]).srcId << "," << ((Edge)edges[j]).desId << "," << ((Edge)edges[j]).weight <<  ")" << std::endl;
			}
		}
		std::cout << std::endl << "----------------------" << std::endl << std::endl;
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
			std::cout << std::endl;
		}
		std::cout << std::endl << "----------------------" << std::endl << std::endl;
	}



	//------- METHOD : parallel for each cube, get the set of edges & create the submerge tree. Globally combine them

	//---------------- METHOD - Local functions

	// locally, get intra-cube edges for each cube & create the local merge trees
	void localStep()
	{
		int cubes = ignoreEmptyCubes ? cubesNonEmpty : numOfCubes; // get the cubes which should be considered

		// resize vectors necessary for merge tree construction
		tmpNxt.resize(numOfCubes);
		tmpFree.resize(numOfParticles);
		nodes.resize(numOfParticles);
		nodesTmp1.resize(numOfParticles);

    struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff1, diff2, diff3, diff4, diff5, diff6;
    gettimeofday(&begin, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				initNodes(thrust::raw_pointer_cast(&*nodes.begin()),
									thrust::raw_pointer_cast(&*nodesTmp1.begin()),
									thrust::raw_pointer_cast(&*tmpFree.begin()),
									numOfParticles));
    gettimeofday(&mid1, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				initNxt(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
								thrust::raw_pointer_cast(&*tmpNxt.begin()),
								thrust::raw_pointer_cast(&*tmpFree.begin()),
								numOfCubes, numOfParticles));
    gettimeofday(&mid2, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				createSubMergeTree(thrust::raw_pointer_cast(&*nodes.begin()),
													 thrust::raw_pointer_cast(&*nodesTmp1.begin()),
													 thrust::raw_pointer_cast(&*particleId.begin()),
													 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
												 	 thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
													 thrust::raw_pointer_cast(&*tmpNxt.begin()),
													 thrust::raw_pointer_cast(&*tmpFree.begin()),
													 min_ll));
		gettimeofday(&mid3, 0);
		initEdgeArrays();  				// init arrays needed for storing edges
		gettimeofday(&mid4, 0);
		getEdgesPerCube(); 				// for each cube, get the set of edges
		gettimeofday(&mid5, 0);
		sortCubeIDByParticles();	// sort cube ids by particle id
    gettimeofday(&end, 0);

		#ifdef TEST
			outputMergeTreeDetails("The local merge trees.."); // output merge tree details
			outputEdgeDetails("Edges to be considered in the global step.."); // output edge details
		#endif

    std::cout << std::endl;
	  std::cout << "'localStep' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed0: " << seconds1 << " s for initNodes " << std::endl << std::flush;
		timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed1: " << seconds2 << " s for initNxt " << std::endl << std::flush;
		timersub(&mid3, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed2: " << seconds3 << " s for createSubMergeTree " << std::endl << std::flush;   
		timersub(&mid4, &mid3, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed3: " << seconds4 << " s for initEdgeArrays " << std::endl << std::flush;
		timersub(&mid5, &mid4, &diff5);
    float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
    std::cout << "Time elapsed4: " << seconds5 << " s for getEdgesPerCube " << std::endl << std::flush;
		timersub(&end, &mid5, &diff6);
    float seconds6 = diff6.tv_sec + 1.0E-6*diff6.tv_usec;
    std::cout << "Time elapsed5: " << seconds6 << " s for sortCubeIDByParticles " << std::endl << std::flush;
	}

	// for each particle, init the nodes array with node id, halo id, count & set the tmpFree array with its initial nxt free id
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

		int   numOfCubes, numOfParticles;

		__host__ __device__
		initNxt(int *particleSizeOfCubes, int *particleStartOfCubes, 
			int *tmpNxt, int *tmpFree,
			int numOfCubes, int numOfParticles) : 
			particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes),			
			tmpNxt(tmpNxt), tmpFree(tmpFree),
			numOfCubes(numOfCubes), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {
			int start = particleStartOfCubes[i];
			int size  = particleSizeOfCubes[i];

		  tmpNxt[i] = (size>0) ? start : -1;

			tmpFree[start+size-1] = -1;
    }
 	};

	// create the submerge tree for each cube
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

  // for each cube, init arrays needed for storing edges
	void initEdgeArrays()
	{
		int cubes = ignoreEmptyCubes ? cubesNonEmpty : numOfCubes; // get the cubes which should be considered

		// for each cube, set the space required for storing edges
		edgeSizeOfCubes.resize(cubes);
		edgeStartOfCubes.resize(cubes);

		// for each vube, get the details of how many neighbors should be checked 
		side = (1 + std::ceil(max_ll/cubeLen)*2);
		size = side*side*side;
		ite = (size-1)/2;

		std::cout << std::endl << "side " << side << " cubeSize " << size << " ite " << ite << std::endl;
		std::cout << cubesEmpty << " of " << numOfCubes << " cubes are empty. (" << (((double)cubesEmpty*100)/(double)numOfCubes) << "%)" << std::endl;

//------------ TODO : this needs to be changed to allocate just the amount of space necessary for the edges, nothing more
		thrust::fill(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes, ite);
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes, edgeStartOfCubes.begin());
		numOfEdges = edgeStartOfCubes[cubes-1] + edgeSizeOfCubes[cubes-1]; // size of edges array 
//------------

		std::cout << std::endl << "numOfEdges before " << numOfEdges << std::endl;

		// init edge arrays
		edges.resize(numOfEdges);
	}

//------------ TODO : Look at whether I am doing the right thing in chunking
	// for each cube, get the set of edges by running them in chunks of cubes
	void getEdgesPerCube()
	{	
		int cubes = ignoreEmptyCubes ? cubesNonEmpty : numOfCubes; // get the cubes which should be considered

		thrust::device_vector<int>::iterator maxSize = thrust::max_element(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes);

		startOfChunks.resize(cubes);
		sizeOfChunks.resize(cubes);

		thrust::fill(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
		thrust::inclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes, sizeOfChunks.begin());
		/*
		std::cout << "particleSizeOfCubes	 "; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "inclusive_scan	 "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		*/
		thrust::copy_if(CountingIterator(0), CountingIterator(0)+cubes, startOfChunks.begin(),
				isStartOfChunks(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
												thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
										 		*maxSize, cubes));	
		int chunks = cubes - thrust::count(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
				setSizeOfChunks(thrust::raw_pointer_cast(&*startOfChunks.begin()),
												thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
												chunks, cubes));

		std::cout << "maxSize " << *maxSize << " chunks " << chunks << std::endl;
		/*
		std::cout << "startOfChunks	 "; thrust::copy(startOfChunks.begin(), startOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "sizeOfChunks	 "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;

		std::cout << "amountOfChunks	 ";
		for(int i=0; i<chunks; i++)
		{
			int count = 0;
			for(int j=startOfChunks[i]; j<startOfChunks[i]+sizeOfChunks[i]; j++)
				count += particleSizeOfCubes[j];
			std::cout << count << " ";
		}
		std::cout << std::endl << std::endl;
		*/
		thrust::fill(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes,0);		
		
		thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
				getEdges(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
								 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*startOfChunks.begin()),
								 thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
								 thrust::raw_pointer_cast(&*cubeMapping.begin()),
								 thrust::raw_pointer_cast(&*cubeMappingInv.begin()),
								 thrust::raw_pointer_cast(&*inputX.begin()),
								 thrust::raw_pointer_cast(&*inputY.begin()),
								 thrust::raw_pointer_cast(&*inputZ.begin()),
								 thrust::raw_pointer_cast(&*particleId.begin()),
								 max_ll, min_ll, ite, cubesInX, cubesInY, cubesInZ, side,
								 thrust::raw_pointer_cast(&*edges.begin()),
								 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));		
/*
		for(int i=0; i<chunks; i++)
		{
	    struct timeval begin, end, diff;
  		gettimeofday(&begin, 0);
			thrust::for_each(CountingIterator(i), CountingIterator(i)+1,
				getEdges(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
								 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*startOfChunks.begin()),
								 thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
								 thrust::raw_pointer_cast(&*cubeMapping.begin()),
								 thrust::raw_pointer_cast(&*cubeMappingInv.begin()),
								 thrust::raw_pointer_cast(&*inputX.begin()),
								 thrust::raw_pointer_cast(&*inputY.begin()),
								 thrust::raw_pointer_cast(&*inputZ.begin()),
								 thrust::raw_pointer_cast(&*particleId.begin()),
								 max_ll, min_ll, ite, cubesInX, cubesInY, cubesInZ, side,
								 thrust::raw_pointer_cast(&*edges.begin()),
								 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));
			gettimeofday(&end, 0);

			timersub(&end, &begin, &diff);
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    std::cout << "Time elapsed....: " << seconds << " s for chunk " << i << std::endl << std::flush;		
		}
*/
		numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes); // set the correct number of edges

		std::cout << "numOfEdges after " << numOfEdges << std::endl;
	}

	// check whether this cube is the start of a chunk
  struct isStartOfChunks : public thrust::unary_function<int, void>
  {
		int size, max;	
		int *sizeOfChunks, *particleSizeOfCubes;

		__host__ __device__
		isStartOfChunks(int *particleSizeOfCubes, int *sizeOfChunks, int max, int size) : 
			particleSizeOfCubes(particleSizeOfCubes), sizeOfChunks(sizeOfChunks), 
			max(max), size(size) {}

    __host__ __device__
    bool operator()(int i)
    {
			if(i==0) return true;
	
			int a = sizeOfChunks[i] / max;	int b = sizeOfChunks[i-1] / max;	
			int d = sizeOfChunks[i] % max;	int e = sizeOfChunks[i-1] % max;

			if((a!=b && d==0 && e==0) ||(a!=b && d!=0) || (a==b && e==0)) return true;

			return false;
    }
 	};

	// set the size of cube
  struct setSizeOfChunks : public thrust::unary_function<int, void>
  {
		int chunks, cubes;	
		int *startOfChunks, *sizeOfChunks;

		__host__ __device__
		setSizeOfChunks(int *startOfChunks, int *sizeOfChunks, int chunks, int cubes) : 
			startOfChunks(startOfChunks), sizeOfChunks(sizeOfChunks), chunks(chunks), cubes(cubes) {}

    __host__ __device__
    void operator()(int i)
    {
			if(i==chunks-1) sizeOfChunks[i] = cubes - startOfChunks[i];
			else 						sizeOfChunks[i] = startOfChunks[i+1] - startOfChunks[i];
    }
 	};

//------------ TODO : try to do the edge calculation parallely for the number of iterations as well
	// for each cube, get the set of edges after comparing
	struct getEdges : public thrust::unary_function<int, void>
	{
		int    ite;
		float  max_ll, min_ll;
		float *inputX, *inputY, *inputZ;

		int   *startOfChunks, *sizeOfChunks;
		int   *cubeMapping, *cubeMappingInv;
		int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;

		int    side;
		int    cubesInX, cubesInY, cubesInZ;
		
		Edge  *edges;
		int   *edgeStartOfCubes, *edgeSizeOfCubes;

		__host__ __device__
		getEdges(int *particleStartOfCubes, int *particleSizeOfCubes, 
				int *startOfChunks, int *sizeOfChunks,
				int *cubeMapping, int *cubeMappingInv,
				float *inputX, float *inputY, float *inputZ,
				int *particleId, float max_ll, float min_ll, int ite,
				int cubesInX, int cubesInY, int cubesInZ, int side, 
				Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes) :
 				particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
				startOfChunks(startOfChunks), sizeOfChunks(sizeOfChunks),
				cubeMapping(cubeMapping), cubeMappingInv(cubeMappingInv), 
			 	inputX(inputX), inputY(inputY), inputZ(inputZ),
				particleId(particleId), max_ll(max_ll), min_ll(min_ll), ite(ite), 
				cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), side(side),
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

		__host__ __device__
		void operator()(int i)
		{	
			for(int l=startOfChunks[i]; l<startOfChunks[i]+sizeOfChunks[i]; l++)
			{		
				int i_mapped = cubeMapping[l];

				// get x,y,z coordinates for the cube
				int tmp = i_mapped % (cubesInX*cubesInY);
				int z = i_mapped / (cubesInX*cubesInY);
				int y = tmp / cubesInX;
				int x = tmp % cubesInX;

				int len = (side-1)/2;

				for(int num=0; num<ite; num++)
				{
					int tmp1 = num % (side*side);
					int z1 = num / (side*side);
					int y1 = tmp1 / side;
					int x1 = tmp1 % side;		

					// get x,y,z coordinates for the current cube 
					int currentX = x - len + x1;
					int currentY = y - len + y1;
					int currentZ = z - len + z1;

					int cube_mapped = -1, cube = -1;
					if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
					{
						cube_mapped = (currentZ*(cubesInY*cubesInX) + currentY*cubesInX + currentX);
						cube = cubeMappingInv[cube_mapped];
					}

					if(cube_mapped==-1 || particleSizeOfCubes[i]==0 || particleSizeOfCubes[cube]==0) continue;

					Edge  e;
					float dist_min = max_ll+1;					

					// for each particle in this cube
					for(int j=particleStartOfCubes[l]; j<particleStartOfCubes[l]+particleSizeOfCubes[l]; j++)
					{
						int pId_j = particleId[j];

						// compare with particles in neighboring cube
						for(int k=particleStartOfCubes[cube]; k<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; k++)
						{
							int pId_k = particleId[k];

							float xd, yd, zd;
							xd = (inputX[pId_j]-inputX[pId_k]);  if (xd < 0.0f) xd = -xd;
							yd = (inputY[pId_j]-inputY[pId_k]);  if (yd < 0.0f) yd = -yd;
							zd = (inputZ[pId_j]-inputZ[pId_k]);  if (zd < 0.0f) zd = -zd;
																				                                                                                                                   						
							if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
							{
								float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);

								if(dist <= max_ll && dist < dist_min)
								{
									int srcV = (pId_j <= pId_k) ? pId_j : pId_k;
									int desV = (srcV == pId_k)  ? pId_j : pId_k;							

									dist_min = dist;		
									e = Edge(srcV, desV, dist);		

									if(dist_min <= min_ll) goto loop;
								}							
							}			
						}
					}

					// add edge
					loop:
					if(dist_min < max_ll + 1)
					{
						edges[edgeStartOfCubes[l] + edgeSizeOfCubes[l]] = e;
						edgeSizeOfCubes[l]++;
					}
				}
			}
		}
	};




	//---------------- METHOD - Global functions

  // combine two local merge trees, two cubes at a time
	void globalStep()
	{
		int cubes = ignoreEmptyCubes ? cubesNonEmpty : numOfCubes; // get the cubes which should be considered

		int cubesOri = cubes;
		int sizeP = 2;

		// set new number of cubes
		int cubesOld = cubes;
		cubes = (int)std::ceil(((double)cubes/2));

		std::cout << std::endl;

		if(numOfEdges==0) return;
	
		// iteratively combine the cubes two at a time
		while(cubes!=cubesOld && cubes>0)
		{
	    struct timeval begin, end, diff;
	    gettimeofday(&begin, 0);
			thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes, 
				combineFreeLists(thrust::raw_pointer_cast(&*tmpNxt.begin()),
												 thrust::raw_pointer_cast(&*tmpFree.begin()),
												 sizeP, cubesOri));
			thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				combineMergeTrees(thrust::raw_pointer_cast(&*cubeMapping.begin()),
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
													thrust::raw_pointer_cast(&*edges.begin()),
													thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
													thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
													min_ll, max_ll, sizeP, cubesOri, numOfParticles));
			gettimeofday(&end, 0);

      timersub(&end, &begin, &diff);
      float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
      std::cout << "Time elapsed: " << seconds << " s for nonEmptyCubes " << cubes;

			numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubesOri);
			std::cout << " numOfEdges after " << numOfEdges << std::endl;

			// set new number of cubes & sizeP
			sizeP *= 2;
			cubesOld = cubes;
			cubes = (int)std::ceil(((double)cubes/2));
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

				k += sizeP/2;
				if(k<cubeEnd) tmpFree[nxt] = tmpNxt[k];
			}
		}
	};

	// combine two local merge trees
	struct combineMergeTrees : public thrust::unary_function<int, void>
  {
    float  min_ll, max_ll;
		int    sizeP, numOfCubesOri, numOfParticles;

    int   *cubeId, *cubeMapping;
		int   *tmpNxt, *tmpFree;

    int   *particleStartOfCubes, *particleSizeOfCubes;

		float *inputX, *inputY, *inputZ;

		Edge *edges;
		int  *edgeStartOfCubes, *edgeSizeOfCubes;

    Node  *nodes, *nodesTmp1;

    __host__ __device__
    combineMergeTrees(int *cubeMapping, int *particleSizeOfCubes, int *particleStartOfCubes, 
				float *inputX, float *inputY, float *inputZ,
        int *cubeId, Node *nodesTmp1, Node *nodes, int *tmpNxt, int *tmpFree, 
				Edge *edges, int *edgeStartOfCubes, int *edgeSizeOfCubes,
				float min_ll, float max_ll, int sizeP, int numOfCubesOri, int numOfParticles) :
        cubeMapping(cubeMapping), particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes), 
				inputX(inputX), inputY(inputY), inputZ(inputZ),
        cubeId(cubeId), nodesTmp1(nodesTmp1), nodes(nodes),	tmpNxt(tmpNxt), tmpFree(tmpFree), 
				edges(edges), edgeStartOfCubes(edgeStartOfCubes), edgeSizeOfCubes(edgeSizeOfCubes),
				min_ll(min_ll), max_ll(max_ll), sizeP(sizeP), numOfCubesOri(numOfCubesOri), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {	
			int cubeStart = sizeP*i;
			int cubeEnd   = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

			int cubeStartM = cubeMapping[cubeStart];
			int cubeEndM   = cubeMapping[cubeEnd-1];

			// get the edges
			for(int k=cubeStart; k<cubeEnd; k++)
			{	
				int size = 0;
				for(int j=edgeStartOfCubes[k]; j<edgeStartOfCubes[k]+edgeSizeOfCubes[k]; j++)
				{
					Edge e = ((Edge)edges[j]);

					if(!(cubeId[e.srcId]>=cubeStartM && cubeId[e.srcId]<=cubeEndM) || 
						 !(cubeId[e.desId]>=cubeStartM && cubeId[e.desId]<=cubeEndM))
					{
						edges[edgeStartOfCubes[k] + size++] = e;
						continue;
					}

					// use this edge (e), to combine the merge trees
					//-----------------------------------------------------

					Node *src = &nodes[e.srcId];
					Node *des = &nodes[e.desId];

					float weight = (e.weight < min_ll) ? min_ll : e.weight; 

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
						if(child && child->value==src->value && child->haloId==src->haloId) 
							srcTmp->childS = child->sibling;
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
//						else
//							std::cout << "***no Free item .... this shouldnt happen*** " << cubeStart << " " << e.weight << " " << min_ll << " " << std::endl;
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

				edgeSizeOfCubes[k] = size;
			}
		}		
	};	
};

}

#endif

