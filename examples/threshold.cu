/*
 * threshold.cu
 *
 *  Created on: Sep 21, 2011
 *      Author: ollie
 */

#include <thrust/device_vector.h>
#if 1
#include <GL/glut.h>
#include <piston/sphere.h>
#include <piston/threshold_geometry.h>

//#define SPACE  thrust::host_space_tag
#define SPACE thrust::detail::default_device_space_tag

using namespace piston;
static const int GRID_SIZE = 256;

#if 1
template <typename IndexType, typename ValueType>
struct height_field : public piston::image3d<IndexType, ValueType, SPACE>
{
    struct height_functor : public piston::implicit_function3d<IndexType, ValueType> {
	typedef piston::implicit_function3d<IndexType, ValueType> Parent;
	typedef typename Parent::InputType InputType;

	__host__ __device__
	ValueType operator()(InputType pos) const {
	    return thrust::get<2>(pos);
	};
    };

    typedef piston::image3d<IndexType, ValueType, SPACE> Parent;

    typedef thrust::transform_iterator<height_functor,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
    PointDataIterator iter;

    height_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	iter(this->grid_coordinates_iterator,
	     height_functor()){}

    PointDataIterator point_data_begin() {
	return iter;
    }

    PointDataIterator point_data_end() {
	return iter + this->NPoints;
    }
};
#endif

#if 0

template <typename IndexType, typename ValueType>
struct sphere_field : public piston::image3d<IndexType, ValueType, SPACE>
{
    typedef piston::image3d<IndexType, ValueType, SPACE> Parent;

    typedef thrust::transform_iterator<piston::sphere<IndexType, ValueType>,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
    PointDataIterator iter;

    sphere_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	iter(this->grid_coordinates_iterator,
	     piston::sphere<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)){}

    PointDataIterator point_data_begin() {
	return iter;
    }

    PointDataIterator point_data_end() {
	return iter+this->NPoints;
    }
};

#else

template <typename IndexType, typename ValueType>
struct sphere_field : public piston::image3d<IndexType, ValueType, SPACE>
{
    typedef piston::image3d<IndexType, ValueType, SPACE> Parent;

//    typedef thrust::host_vector<thrust::tuple<IndexType, IndexType, IndexType> > GridCoordinatesContainer;
    typedef typename choose_container<typename Parent::CountingIterator, thrust::tuple<IndexType, IndexType, IndexType> >::type GridCoordinatesContainer;
    GridCoordinatesContainer grid_coordinates_vector;
    typedef typename GridCoordinatesContainer::iterator GridCoordinatesIterator;
    GridCoordinatesIterator  grid_coordinates_iterator;

//    typedef thrust::host_vector<ValueType> PointDataContainer;
    typedef typename choose_container<typename Parent::CountingIterator, ValueType>::type PointDataContainer;
    PointDataContainer point_data_vector;
    typedef typename PointDataContainer::iterator PointDataIterator;
    PointDataIterator point_data_iterator;

    sphere_field(int xdim, int ydim, int zdim) :
	Parent(xdim, ydim, zdim),
	grid_coordinates_vector(Parent::grid_coordinates_begin(), Parent::grid_coordinates_end()),
	grid_coordinates_iterator(grid_coordinates_vector.begin()),
	point_data_vector(thrust::make_transform_iterator(grid_coordinates_iterator, sphere<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1)),
	                  thrust::make_transform_iterator(grid_coordinates_iterator, sphere<IndexType, ValueType>(xdim/2, ydim/2, zdim/2, 1))+this->NPoints),
	point_data_iterator(point_data_vector.begin()) {}

    GridCoordinatesIterator grid_coordinates_begin() {
	return grid_coordinates_iterator;
    }
    GridCoordinatesIterator grid_coordinates_end() {
	return grid_coordinates_iterator+this->NPoints;
    }

    PointDataIterator point_data_begin() {
	return point_data_iterator;
    }
    PointDataIterator point_data_end() {
	return point_data_iterator+this->NPoints;
    }
};

#endif

struct threshold_between : thrust::unary_function<float, bool>
{
    float min_value;
    float max_value;

    threshold_between(float min_value, float max_value) :
	min_value(min_value), max_value(max_value) {}

    __host__ __device__
    bool operator() (float val) const {
	return (min_value <= val) && (val <= max_value);
    }
};

struct print_float4 : public thrust::unary_function<float4, void>
{
	__host__ __device__
	void operator() (float4 p) {
//	    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
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


threshold_geometry<sphere_field<int, float> > *threshold_p;

template <typename ValueType>
struct color_map : thrust::unary_function<ValueType, float4>
{
    const ValueType min;
    const ValueType max;

    __host__ __device__
    color_map(ValueType min, ValueType max) :
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

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (wireframe) {
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
	glPolygonMode(GL_BACK, GL_FILL);
	glPolygonMode(GL_FRONT, GL_FILL);
    }

    // set view matrix for 3D scene
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPushMatrix();

    glRotatef(rotate.x, 1.0, 0.0, 0.0);
    glRotatef(rotate.y, 0.0, 1.0, 0.0);
    glTranslatef(-(GRID_SIZE-1)/2, -(GRID_SIZE-1)/2, -(GRID_SIZE-1)/2);
    glTranslatef(translate.x, translate.y, translate.z);

    thrust::host_vector<float4> vertices(thrust::make_transform_iterator(threshold_p->vertices_begin(), tuple2float4()),
                                         thrust::make_transform_iterator(threshold_p->vertices_end(),   tuple2float4()));
    thrust::host_vector<float4> colors(thrust::make_transform_iterator(threshold_p->scalars_begin(), color_map<float>(4.0f, 256.0f)),
                                       thrust::make_transform_iterator(threshold_p->scalars_end(),  color_map<float>(4.0f, 256.0f)));

//    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

//    glNormalPointer(GL_FLOAT, 0, &normals[0]);
    glColorPointer(4, GL_FLOAT, 0, &colors[0]);
    glVertexPointer(4, GL_FLOAT, 0, &vertices[0]);
    glDrawArrays(GL_QUADS, 0, vertices.size());

    // set view matrix for 2D message
    // TBD
    glutSwapBuffers();
}

void idle()
{
    if (animate) {
//	isovalue += delta;
//	if (isovalue > maxiso)
//	    delta = -0.05;
//	if (isovalue < miniso)
//	    delta = 0.05;
//	glutPostRedisplay();
    }
}

void initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Marching Cube");

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    // good old-fashioned fixed function lighting
//    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float white[] = { 0.8, 0.8, 0.8, 1.0 };
//    float ambient[] = { 0.5, 0.0, 0.0, 1.0 };
//    float diffuse[] = { 0.5, 0.0, 0.0, 1.0 };
    float lightPos[] = { 100.0, 100.0, -100.0, 1.0 };

//    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
//    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
//    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100);

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
//    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
//    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
//    glEnable(GL_NORMALIZE);
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
//    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

//    glutReshapeFunc( reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
//    glutIdleFunc(idle);
    glutMainLoop();
}

#endif

int main(int argc, char *argv[])
{

    sphere_field<int, float> scalar_field(GRID_SIZE, GRID_SIZE, GRID_SIZE);
//    thrust::copy(scalar_field.point_data_begin(), scalar_field.point_data_end(), std::ostream_iterator<float>(std::cout, " "));
//    std::cout << std::endl;

    threshold_geometry<sphere_field<int, float> > threshold(scalar_field, 4, 1600);
//    for (int i = 0; i < 10; i++)
	threshold();

//    threshold_p = &threshold;

//    initGL(argc, argv);

    return 0;
}
