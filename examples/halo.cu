
using namespace std;

#include <piston/halo_kd.h>
#include <piston/halo_vtk.h>
//-------

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

using namespace piston;

int main(int argc, char* argv[])
{
  halo *halo;

  float linkLength;
  int   particleSize, rL, np;

  linkLength   = 0.2;
  particleSize = 100;
  np = 256;
  rL = 100;

  int n = 1; // if you want a fraction of the file to load, use this.. 1/n
  char filename[1024];
  sprintf(filename, "%s/sub-24474", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
  std::cout << filename << std::endl;

  halo = new halo_kd(filename, "csv", n, np, rL); // 214414, 24474
  (*halo)(linkLength, particleSize);

  std::cout << "VTK reference result" << std::endl;

  halo = new halo_vtk(filename, "csv", n, np, rL);
  (*halo)(linkLength, particleSize);

  return 0;
}
