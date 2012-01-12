/*
 * simple_opengl.cu
 *
 *  Created on: Oct 7, 2011
 *      Author: ollie
 */

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <GL/glut.h>

#include <thrust/device_vector.h>
#include <thrust/distance.h>

#include <piston/implicit_function.h>
#include <piston/cutil_math.h>
#include <piston/image2d.h>

//#define SPACE  thrust::host_space_tag
#define SPACE thrust::detail::default_device_space_tag

using namespace piston;
static const int GRID_SIZE = 4;

struct sine_wave : public piston::image2d<int, float4, SPACE>
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
    typedef thrust::transform_iterator<sine_functor,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
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
    __host__ __device__
    void operator() (thrust::tuple<int, int> pos) {
	std::cout << "(" << thrust::get<0>(pos) << ", "
		         << thrust::get<1>(pos) << ")" << std::endl;
    }
};

struct print_float4 : public thrust::unary_function<float4, void>
{
	__host__ __device__
	void operator() (float4 p) {
	    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ", " << p.w <<")" << std::endl;
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
    sine_wave field(GRID_SIZE, GRID_SIZE);

    thrust::host_vector<thrust::tuple<int, int> > position(field.grid_coordinates_begin(), field.grid_coordinates_end());
    thrust::for_each(position.begin(), position.end(), print_tuple2());
    thrust::host_vector<float4> points(field.point_data_begin(), field.point_data_end());
    thrust::for_each(points.begin(), points.end(), print_float4());

    field.resize(2*GRID_SIZE, 2*GRID_SIZE);
    position.resize(thrust::distance(field.grid_coordinates_begin(), field.grid_coordinates_end()));
    points.resize(thrust::distance(field.grid_coordinates_begin(), field.grid_coordinates_end()));

    thrust::copy(field.grid_coordinates_begin(), field.grid_coordinates_end(), position.begin());
    thrust::for_each(position.begin(), position.end(), print_tuple2());
    thrust::copy(field.point_data_begin(), field.point_data_end(), points.begin());
    thrust::for_each(points.begin(), points.end(), print_float4());

    return 0;
}
#endif
