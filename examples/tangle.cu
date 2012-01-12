/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.
Copyright 2011. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.

If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
·         Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
·         Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
          materials provided with the distribution.
·         Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used
          to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>

#include <cutil_math.h>
#include <piston/choose_container.h>

#define SPACE thrust::detail::default_device_space_tag
using namespace piston;

template <typename ValueType>
struct color_map : thrust::unary_function<ValueType, float4>
{
    const ValueType min;
    const ValueType max;

    __host__ __device__
    color_map(ValueType min, ValueType max, bool reversed=false) :
	min(min), max(max) {}

    __host__ __device__
    float4 operator()(ValueType val) {
	// HSV rainbow for height field, stolen form Manta
	const float V = 0.7f, S = 1.0f;
	float H = (1.0f - static_cast<float> (val) / (max - min));

	if (H < 0.0f)
	    H = 0.0f;
	else if (H > 1.0f)
	    H = 1.0f;
	H *= 4.0f;

	float i = floor(H);
	float f = H - i;

	float p = V * (1.0 - S);
	float q = V * (1.0 - S * f);
	float t = V * (1.0 - S * (1 - f));

	float R, G, B;
	if (i == 0.0) {
	    R = V;
	    G = t;
	    B = p;
	} else if (i == 1.0) {
	    R = q;
	    G = V;
	    B = p;
	} else if (i == 2.0) {
	    R = p;
	    G = V;
	    B = t;
	} else if (i == 3.0) {
	    R = p;
	    G = q;
	    B = V;
	} else if (i == 4.0) {
	    R = t;
	    G = p;
	    B = V;
	} else {
	    // i == 5.0
	    R = V;
	    G = p;
	    B = q;
	}
	return make_float4(R, G, B, 1.0);
    }
};

#include <piston/implicit_function.h>
#include <piston/image3d.h>
#include <piston/marching_cube.h>
#include <sys/time.h>
#include <stdio.h>

static const int GRID_SIZE = 256;

template <typename IndexType, typename ValueType, typename Space>
struct tangle_field : public piston::image3d<IndexType, ValueType, Space>
{
    typedef piston::image3d<IndexType, ValueType, Space> Parent;

//    typedef typename detail::choose_container<typename Parent::CountingIterator, thrust::tuple<IndexType, IndexType, IndexType> >::type GridCoordinatesContainer;
//    GridCoordinatesContainer grid_coordinates_vector;
//    typedef typename GridCoordinatesContainer::iterator GridCoordinatesIterator;

    typedef typename detail::choose_container<typename Parent::CountingIterator, ValueType>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;

    struct tangle_functor : public piston::implicit_function3d<IndexType, ValueType>
    {
	typedef piston::implicit_function3d<IndexType, ValueType> Parent;
	typedef typename Parent::InputType InputType;

        const float xscale;
        const float yscale;
        const float zscale;

        tangle_functor(IndexType xdim, IndexType ydim, IndexType zdim) :
            xscale(2.0f/(xdim - 1.0f)),
            yscale(2.0f/(ydim - 1.0f)),
            zscale(2.0f/(zdim - 1.0f)) {}

        __host__ __device__
        float operator()(InputType pos) const {
            // scale and shift such that x, y, z <- [-1,1]
            const float x = 3.0f*(thrust::get<0>(pos)*xscale - 1.0f);
            const float y = 3.0f*(thrust::get<1>(pos)*yscale - 1.0f);
            const float z = 3.0f*(thrust::get<2>(pos)*zscale - 1.0f);

            const float v = (x*x*x*x - 5.0f*x*x +y*y*y*y - 5.0f*y*y +z*z*z*z - 5.0f*z*z + 11.8f) * 0.2f + 0.5f;

            return v;
        }
    };

    tangle_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
//	grid_coordinates_vector(Parent::grid_coordinates_begin(), Parent::grid_coordinates_end()),
//	point_data_vector(thrust::make_transform_iterator(grid_coordinates_vector.begin(), tangle_functor(xdim, ydim, zdim)),
//	                  thrust::make_transform_iterator(grid_coordinates_vector.end(),   tangle_functor(xdim, ydim, zdim)))
//	grid_coordinates_vector(Parent::grid_coordinates_begin(), Parent::grid_coordinates_end()),
	point_data_vector(thrust::make_transform_iterator(Parent::grid_coordinates_begin(), tangle_functor(xdim, ydim, zdim)),
	                  thrust::make_transform_iterator(Parent::grid_coordinates_end(),   tangle_functor(xdim, ydim, zdim)))
	                  {}

    void resize(int xdim, int ydim, int zdim) {
	Parent::resize(xdim, ydim, zdim);
	point_data_vector.resize(this->NPoints);
//	point_data_vector.assign(thrust::make_transform_iterator(grid_coordinates_vector.begin(), tangle_functor(xdim, ydim, zdim)),
//	                         thrust::make_transform_iterator(grid_coordinates_vector.end(),   tangle_functor(zdim, ydim, zdim)));
    }

//    GridCoordinatesIterator grid_coordinates_begin() {
//	return grid_coordinates_vector.begin();
//    }
//    GridCoordinatesIterator grid_coordinates_end() {
//	return grid_coordinates_vector.end();
//    }

    PointDataIterator point_data_begin() {
	return point_data_vector.begin();
    }
    PointDataIterator point_data_end() {
	return point_data_vector.end();
    }
};



int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float3 rotate = make_float3(0, 0, 0.0);
float3 translate = make_float3(0.0, 0.0, 0.0);

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
	mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
	mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx = x - mouse_old_x;
    float dy = y - mouse_old_y;

    if (mouse_buttons==1) {
	rotate.x += dy * 0.2;
	rotate.y += dx * 0.2;
    } else if (mouse_buttons==2) {
	translate.x += dx * 0.01;
	translate.y -= dy * 0.01;
    } else if (mouse_buttons==4) {
	translate.z += dy * 0.1;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

bool wireframe = false;
bool animate = true;
void keyboard( unsigned char key, int x, int y )
{
    switch (key) {
    case 'w':
	wireframe = !wireframe;
	break;
    case 'a':
	animate = !animate;
	break;
    }
}

struct tuple2float4 : thrust::unary_function<thrust::tuple<int, int, int>, float4>
{
	__host__ __device__
	float4 operator()(thrust::tuple<int, int, int> xyz) {
	    return make_float4((float) thrust::get<0>(xyz),
	                       (float) thrust::get<1>(xyz),
	                       (float) thrust::get<2>(xyz),
	                       1.0f);
	}
};


marching_cube<tangle_field<int, float, SPACE>, tangle_field<int, float, SPACE> > *isosurface_p;

GLuint quads_vbo[3];
struct cudaGraphicsResource *quads_pos_res, *quads_normal_res, *quads_color_res;
unsigned int buffer_size;

void create_vbo()
{
    glGenBuffers(3, quads_vbo);
    int error;

    std::cout << "number of vertices: " << thrust::distance(isosurface_p->vertices_begin(), isosurface_p->vertices_end()) << std::endl;
    buffer_size = thrust::distance(isosurface_p->vertices_begin(), isosurface_p->vertices_end())* sizeof(float4);

    // initialize vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_pos_res, quads_vbo[0],
                                              cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register pos buffer cuda error: " << error << "\n";
    }

    // initialize vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_normal_res, quads_vbo[1],
                                              cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register normal buffer cuda error: " << error << "\n";
    }

    // initialize color buffer object
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // register this buffer object with CUDA
    if ((error = cudaGraphicsGLRegisterBuffer(&quads_color_res, quads_vbo[2],
                                     cudaGraphicsMapFlagsWriteDiscard)) != cudaSuccess) {
	std::cout << "register color buffer cuda error: " << error << "\n";
    }
}

int frame_count = 0;
float seconds = 0.0f;
bool new_frame = false;

void display()
{
    struct timeval begin, end, diff;
    gettimeofday(&begin, 0);

    (*isosurface_p)();
#if 1
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (wireframe) {
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
//	glPolygonMode(GL_BACK, GL_LINE);
//	glPolygonMode(GL_FRONT, GL_FILL);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // set view matrix for 3D scene
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPushMatrix();

    glRotatef(rotate.x, 1.0, 0.0, 0.0);
    glRotatef(rotate.y, 0.0, 1.0, 0.0);
    glTranslatef(-(GRID_SIZE-1)/2, -(GRID_SIZE-1)/2, -(GRID_SIZE-1)/2);
    glTranslatef(translate.x, translate.y, translate.z);

    float4 *raw_ptr;
    size_t num_bytes;

    if (new_frame) {
	cudaGraphicsMapResources(1, &quads_pos_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&raw_ptr, &num_bytes, quads_pos_res);
	//    thrust::copy(thrust::make_transform_iterator(isosurface_p->vertices_begin(), tuple2float4()),
	//                 thrust::make_transform_iterator(isosurface_p->vertices_end(),   tuple2float4()),
	//                 thrust::device_ptr<float4>(raw_ptr));
	thrust::copy(isosurface_p->vertices_begin(),
	             isosurface_p->vertices_end(),
	             thrust::device_ptr<float4>(raw_ptr));
	cudaGraphicsUnmapResources(1, &quads_pos_res, 0);
    }
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[0]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    float3 *normal;
    if (new_frame) {
	cudaGraphicsMapResources(1, &quads_normal_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&normal, &num_bytes, quads_normal_res);
	thrust::copy(isosurface_p->normals_begin(),
	             isosurface_p->normals_end(),
	             thrust::device_ptr<float3>(normal));
	cudaGraphicsUnmapResources(1, &quads_normal_res, 0);
    }
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[1]);
    glNormalPointer(GL_FLOAT, 0, 0);

    if (new_frame) {
	cudaGraphicsMapResources(1, &quads_color_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&raw_ptr, &num_bytes, quads_color_res);
	thrust::transform(isosurface_p->scalars_begin(), isosurface_p->scalars_end(),
	                  thrust::device_ptr<float4>(raw_ptr),
	                  color_map<float>(0.0f, .9f));
	cudaGraphicsUnmapResources(1, &quads_color_res, 0);
    }
    glBindBuffer(GL_ARRAY_BUFFER, quads_vbo[2]);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glDrawArrays(GL_TRIANGLES, 0, buffer_size/sizeof(float4));

    glutSwapBuffers();
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    new_frame = false;
#endif
    gettimeofday(&end, 0);
    timersub(&end, &begin, &diff);
    frame_count++;
    seconds += diff.tv_sec + 1.0E-6*diff.tv_usec;

    if (frame_count > 10) {
	char title[256];
	sprintf(title, "Tangle, fps: %2.2f", 10.0f/seconds);
	glutSetWindowTitle(title);
	seconds = 0.0f;
	frame_count = 0;
    }

}

void idle()
{
    if (animate) {
//	isovalue += delta;
//	if (isovalue > maxiso)
//	    delta = -0.05;
//	if (isovalue < miniso)
//	    delta = 0.05;
    }
    glutPostRedisplay();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, GRID_SIZE*4.0f);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void timer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(10, timer, 0);
}

void initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Tangle");

    glewInit();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    // good old-fashioned fixed function lighting
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float white[] = { 0.8, 0.8, 0.8, 1.0 };
    float ambient[] = { 0.5, 0.0, 0.0, 1.0 };
    float diffuse[] = { 0.5, 0.0, 0.0, 1.0 };
    float lightPos[] = { 0.0f, 0.0f, 1.0f, 0.0f };

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
//    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
//    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100);

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
//    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
//    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);

    /* Setup the view of the cube. */
    glMatrixMode(GL_PROJECTION);
    gluPerspective( /* field of view in degree */ 60.0,
                    /* aspect ratio */ 1.0,
                    /* Z near */ 1.0, /* Z far */ GRID_SIZE*4.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, GRID_SIZE*1.5,  /* eye is at (0,0, 1.5*GRID_SIZE) */
              0.0, 0.0, 0.0,		/* center is at (0,0,0) */
              0.0, 1.0, 0.0);		/* up is in positive Y direction */
    glPushMatrix();

    // enable vertex and normal arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
}

int main(int argc, char *argv[])
{
    initGL(argc, argv);
    cudaGLSetGLDevice(0);

    tangle_field<int, float, SPACE> tangle(GRID_SIZE, GRID_SIZE, GRID_SIZE);

    marching_cube<tangle_field<int, float, SPACE>,  tangle_field<int, float, SPACE> > isosurface(tangle, tangle, 0.2f);

    isosurface();
    isosurface_p = &isosurface;
    new_frame = true;

    create_vbo();

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);

    glutMainLoop();

    return 0;
}
