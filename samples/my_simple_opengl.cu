/*
 * my_simple_opengl.cu
 *
 *  Created on: Oct 7, 2011, 2/27/2012
 *      Author: ollie/thorp
 */

#include <iostream>


#ifdef __APPLE__
/* Location of some include headers on the Apple systems:
 * /System/Library/Frameworks/FW.framework/Headers
*/
    #include <GL/glew.h>
    //#include <OpenGL/OpenGL.h>
    #include <OpenGL.framework/Headers/OpenGL.h>
    //#include <OpenGL.h>

    //#include <GLUT/glut.h>
    #include <GLUT.framework/Headers/glut.h>
    //#include <glut.h>
//
//
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
    #include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>
#include <thrust/distance.h>

#include <piston/implicit_function.h>
#include <piston/image2d.h>

//#include <piston/cutil_math.h>
#include <piston/piston_math.h>

//#define SPACE  thrust::host_space_tag
#define SPACE thrust::detail::default_device_space_tag

using namespace piston;
static const int GRID_SIZE = 4;

struct sine_wave: public piston::image2d<int, float4, SPACE>
{
	struct sine_functor : public piston::implicit_function2d<int, float4>
	{
		typedef piston::implicit_function2d<int, float4> Parent;
		typedef typename Parent::InputType InputType;

		int xdim;
		int ydim;
		float time;

		sine_functor(int xdim, int ydim, float time) :
			xdim(xdim), ydim(ydim), time(time) {}

		__host__ __device__
		float4 operator()(InputType pos) const {
			unsigned int x = thrust::get<0>(pos);
			unsigned int y = thrust::get<1>(pos);

			// calculate uv coordinates
			float u = x / (float) xdim;
			float v = y / (float) ydim;
			u = u*2.0f - 1.0f;
			v = v*2.0f - 1.0f;

			// calculate simple sine wave pattern
			float freq = 4.0f;

			float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

			// write output vertex
			return make_float4(u, w, v, 1.0f);
		}
	};

	typedef piston::image2d<int, float4, SPACE> Parent;
	typedef thrust::transform_iterator<sine_functor, typename Parent::GridCoordinatesIterator> PointDataIterator;
	float time;
	PointDataIterator point_data_iterator;

	sine_wave(int xdim, int ydim, float time = 0.0f) :
		Parent(xdim, ydim),
		time(time),
		point_data_iterator(this->grid_coordinates_iterator, sine_functor(xdim, ydim, time)){}

	void resize(int xdim, int ydim) {
		Parent::resize(xdim, ydim);
		point_data_iterator = thrust::make_transform_iterator(grid_coordinates_iterator,
				sine_functor(xdim, ydim, time));
	}

	void set_time(float time) {
		this->time = time;
		point_data_iterator = thrust::make_transform_iterator(grid_coordinates_iterator,
				sine_functor(xdim, ydim, time));
	}

	PointDataIterator point_data_begin() {
		return point_data_iterator;
	}

	PointDataIterator point_data_end() {
		return point_data_iterator + this->NPoints;
	}
};


struct print_tuple2 : public thrust::unary_function<thrust::tuple<int, int>, void>
{
	//__host__ __device__
	void operator() (thrust::tuple<int, int> pos) {
		std::cout << "(" << thrust::get<0>(pos) << ", "
				<< thrust::get<1>(pos) << ")" << std::endl;
	}
};


struct print_float4 : public thrust::unary_function<float4, void>
{
	//__host__ __device__
	void operator() (float4 p) {
	    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ", " << p.w <<")" << std::endl;
	}
};

/// Extracts a dimension of the space, x dim
struct extractY : public thrust::unary_function<float4, float>
{
	//__host__ __device__
	float operator() (float4 p) {
	    return( p.y);
	}

};

/// Extracts a dimension of the space, x dim
struct extractX : public thrust::unary_function<float4, float>
{
	//__host__ __device__
	float operator() (float4 p) {
	    return( p.x);
	}

};

/// Extracts a dimension of the space, x dim
struct extractZ : public thrust::unary_function<float4, float>
{
	//__host__ __device__
	float operator() (float4 p) {
	    return( p.z);
	}

};

/// Extracts a dimension of the space, x dim
struct extractW : public thrust::unary_function<float4, float>
{
	//__host__ __device__
	float operator() (float4 p) {
	    return( p.w);
	}

};


#if 0
bool init_gl(void)
{
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, g_window_width, g_window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)g_window_width / (GLfloat) g_window_height, 0.1, 10.0);

    return true;
} // end init_gl

void display(void)
{
    // transform the mesh
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(g_mesh_width * g_mesh_height);

    thrust::transform(first, last,
                      g_vec.begin(),
                      sine_wave(g_mesh_width,g_mesh_height,g_anim));

    // map the vector into GL
    thrust::device_ptr<float4> ptr = &g_vec[0];

    // pass the device_ptr to the allocator's static function map_buffer
    // to map it into GL
    GLuint buffer = gl_vector::allocator_type::map_buffer(ptr);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, g_translate_z);
    glRotatef(g_rotate_x, 1.0, 0.0, 0.0);
    glRotatef(g_rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, g_mesh_width * g_mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

    g_anim += 0.001;

    // unmap the vector from GL
    gl_vector::allocator_type::unmap_buffer(buffer);
} // end display

void mouse(int button, int state, int x, int y)
{
    if(state == GLUT_DOWN)
    {
	g_mouse_buttons |= 1<<button;
    } // end if
    else if(state == GLUT_UP)
    {
	g_mouse_buttons = 0;
    } // end else if

    g_mouse_old_x = x;
    g_mouse_old_y = y;
    glutPostRedisplay();
} // end mouse

void motion(int x, int y)
{
    float dx, dy;
    dx = x - g_mouse_old_x;
    dy = y - g_mouse_old_y;

    if(g_mouse_buttons & 1)
    {
	g_rotate_x += dy * 0.2;
	g_rotate_y += dx * 0.2;
    } // end if
    else if(g_mouse_buttons & 4)
    {
	g_translate_z += dy * 0.01;
    } // end else if

    g_mouse_old_x = x;
    g_mouse_old_y = y;
} // end motion

void keyboard(unsigned char key, int, int)
{
    switch(key)
    {
    // catch 'esc'
    case(27):
	      // deallocate memory
	      g_vec.clear();
    g_vec.shrink_to_fit();
    exit(0);
    default:
	break;
    } // end switch
} // end keyboard

int main(int argc, char** argv)
{
    // Create GL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(g_window_width, g_window_height);
    glutCreateWindow("Thrust/GL interop");

    // initialize GL
    if(!init_gl())
    {
	throw std::runtime_error("Couldn't initialize OpenGL");
    } // end if

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    // resize the vector to fit the mesh
    g_vec.resize(g_mesh_width * g_mesh_height);

    // transform the mesh
    thrust::counting_iterator<int,thrust::device_space_tag> first(0);
    thrust::counting_iterator<int,thrust::device_space_tag> last(g_mesh_width * g_mesh_height);

    thrust::transform(first, last,
                      g_vec.begin(),
                      sine_wave(g_mesh_width,g_mesh_height,g_anim));

    // start rendering mainloop
    glutMainLoop();

    return 0;
} // end main

#else
int main()
{
	// H has storage for 4 integers
	thrust :: host_vector <int > H (4);

	// initialize individual elements
	H [0] = 14;
	H [1] = 20;
	H [2] = 38;
	H [3] = 46;

	// H. size () returns the size of vector H
	std :: cout << "H has size " << H. size () << std :: endl ;

	// print contents of H
	for ( int i = 0; i < H. size (); i ++)
	std :: cout << "H[" << i << "] = " << H[i] << std :: endl ;

	int maxVal = thrust::reduce(H.begin(), H.end(), 0, thrust::maximum<int>());

	std :: cout << "H has max: " << maxVal << std :: endl ;


	// resize H
	H. resize (2) ;
	std :: cout << "H now has size " << H. size () << std :: endl ;

	// print contents of H again
	for ( int i = 0; i < H. size (); i ++)
	std :: cout << "H[" << i << "] = " << H[i] << std :: endl ;

	// Copy host_vector H to device_vector D
	thrust :: device_vector <int > D = H;

	// elements of D can be modified
	D [0] = 99;
	//D [1] = 88;

	// print contents of D
	for ( int i = 0; i < D. size (); i ++)
		std :: cout << "D[" << i << "] = " << D[i] << std :: endl ;

	/** end of thrust sample code.
	*/

	/**
	 * start of basic piston code.
	 */

	// Declare the basic data space.
	sine_wave field(GRID_SIZE, GRID_SIZE);

	// Declare a host vector to represent the coordinates of all items in the field.
	// The coordinates are (x,y) pairs.
    thrust::host_vector
    <thrust::tuple<int, int> >
    position(field.grid_coordinates_begin(), field.grid_coordinates_end());

    // forall coordinates, from the host, get the data from a specific device location and print it out.
    thrust::for_each(position.begin(), position.end(), print_tuple2());
    /* Also works.
	for ( int i = 0; i < position.size (); i ++)
	std :: cout << "position[" << i << "] = " << "(" << thrust::get<0>(position[i]) << ", "
	                                                      				<< thrust::get<1>(position[i]) << ")"<< std :: endl ;
    */


    // Make a host vector with as many entries as there are in the grid.
    // each entry in the vector has for component values (x, y, z, w)
    // init it with the field point data.
    thrust::host_vector<float4> points(field.point_data_begin(), field.point_data_end());

    std::cout << "test2" << std::endl ;

    // print out the points x,y,z,w values
    thrust::for_each(points.begin(), points.end(), print_float4());




    // Double the size of the grid in both dimentions
    field.resize(2*GRID_SIZE, 2*GRID_SIZE);

    // Resize the position vector to the size of the field.
    /*
     * workspace
     *     typedef typename thrust::counting_iterator<IndexType, MemorySpace> CountingIterator;
     *     typedef typename thrust::transform_iterator<grid_coordinates_functor, CountingIterator> GridCoordinatesIterator;
     *
     *     HAVE:
     *     thrust::transform_iterator<grid_coordinates_functor, thrust::counting_iterator<int, thrust::detail::default_device_space_tag>>
     *
     *     NEED:
     *     thrust::host_vector<int>::iterator
     */
    //thrust::transform_iterator<piston::image2d<int, float4, SPACE>::grid_coordinates_functor, thrust::counting_iterator<int, thrust::detail::default_device_space_tag>> iterBegin= field.grid_coordinates_begin();
    //thrust::transform_iterator<piston::image2d<int, float4, SPACE>::grid_coordinates_functor, thrust::counting_iterator<int, thrust::detail::default_device_space_tag>> iterEnd= field.grid_coordinates_end();
    //  unsigned long int foo= thrust::distance(field.image2d::grid_coordinates_begin(), field.image2d::grid_coordinates_end());
    unsigned long int foo= thrust::distance(field.grid_coordinates_begin(), field.grid_coordinates_end());
    position.resize(foo);
    points.resize(thrust::distance(field.piston::image2d<int,float4,SPACE>::grid_coordinates_begin(), field.image2d<int,float4,SPACE>::grid_coordinates_end()));

    std::cout << "test3" << std::endl ;

    thrust::copy(field.grid_coordinates_begin(), field.grid_coordinates_end(), position.begin());
    thrust::for_each(position.begin(), position.end(), print_tuple2());
    thrust::copy(field.point_data_begin(), field.point_data_end(), points.begin());
    thrust::for_each(points.begin(), points.end(), print_float4());




    std::cout << "Test4: Find the bounding box. " << std::endl ;

    /*
     * Find X dimension bounds.
     */
    thrust::host_vector<float> xVal(foo);
    thrust::transform(points.begin(), points.end(), xVal.begin(), extractX());

	// print contents of D
	for ( int i = 0; i < xVal. size (); i ++)
		std :: cout << "xVal[" << i << "] = " << xVal[i] << std :: endl ;

	//Device vector
	thrust::device_vector<float> xVals(foo);

	// Move to GPU
	xVals=xVal;

    float maxX= thrust::reduce(xVals.begin(), xVals.end(), -1.0f, thrust::maximum<float>());
    float minX= thrust::reduce(xVals.begin(), xVals.end(), +1.0f, thrust::minimum<float>());
    std::cout << "Bounds for X axis = { " << minX << "," << maxX  << "}" << std::endl ;


    /*
     * Find Y dimension bounds.
     */
    thrust::host_vector<float> yVals(foo);
    thrust::transform(points.begin(), points.end(), yVals.begin(), extractY());

    float maxY= thrust::reduce(yVals.begin(), yVals.end(), -1.0f, thrust::maximum<float>());
    float minY= thrust::reduce(yVals.begin(), yVals.end(), +1.0f, thrust::minimum<float>());
    std::cout << "Bounds for Y axis = { " << minY << "," << maxY  << "}" << std::endl ;




    /*
     * Find Z dimension bounds.
     */
    thrust::host_vector<float> zVals(foo);
    thrust::transform(points.begin(), points.end(), zVals.begin(), extractZ());

    float maxZ= thrust::reduce(zVals.begin(), zVals.end(), -1.0f, thrust::maximum<float>());
    float minZ= thrust::reduce(zVals.begin(), zVals.end(), +1.0f, thrust::minimum<float>());
    std::cout << "Bounds for Z axis = { " << minZ << "," << maxZ  << "}" << std::endl ;





    /*
     * Find Y dimension bounds.
     */
    thrust::host_vector<float> wVals(foo);
    thrust::transform(points.begin(), points.end(), wVals.begin(), extractW());

    float maxW= thrust::reduce(wVals.begin(), wVals.end(), -1.0f, thrust::maximum<float>());
    float minW= thrust::reduce(wVals.begin(), wVals.end(), +1.0f, thrust::minimum<float>());
    std::cout << "Bounds for W axis = { " << minW << "," << maxW  << "}" << std::endl ;








    return 0;
} // end main
#endif
