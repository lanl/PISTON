/*
 * threshold.cu
 *
 *  Created on: Sep 21, 2011
 *      Author: ollie
 */

#include <GL/glut.h>
#include <piston/sphere.h>
#include <piston/threshold_geometry.h>

static const int GRID_SIZE = 16;

template <typename IndexType, typename ValueType>
struct height_field : public piston::image3d<IndexType, ValueType, thrust::host_space_tag>
{
    struct height_functor : public piston::implicit_function3d<IndexType, ValueType> {
	typedef piston::implicit_function3d<IndexType, ValueType> Parent;
	typedef typename Parent::InputType InputType;

	__host__ __device__
	ValueType operator()(InputType pos) const {
	    return thrust::get<2>(pos);
	};
    };

    typedef piston::image3d<IndexType, ValueType, thrust::host_space_tag> Parent;

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

template <typename IndexType, typename ValueType>
struct sfield : public piston::image3d<IndexType, ValueType, thrust::host_space_tag>
{
    typedef piston::image3d<IndexType, ValueType, thrust::host_space_tag> Parent;

    typedef thrust::transform_iterator<piston::sphere<IndexType, ValueType>,
				       typename Parent::GridCoordinatesIterator> PointDataIterator;
    PointDataIterator iter;

    sfield(int xdim, int ydim, int zdim) :
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
	    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
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


threshold_geometry<sfield<int, float>, threshold_between> *threshold_p;

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

    thrust::host_vector<float4> vertices(threshold_p->verticesBegin(),
                                         threshold_p->verticesEnd());

//    thrust::for_each(vertices.begin(), vertices.end(), print_float4());

    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

//    glNormalPointer(GL_FLOAT, 0, &normals[0]);
//    glColorPointer(3, GL_FLOAT, 0, &normals[0]);
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
    float black[] = { 0.0, 0.0, 0.0, 1.0 };
    float white[] = { 0.8, 0.8, 0.8, 1.0 };
    float ambient[] = { 0.5, 0.0, 0.0, 1.0 };
    float diffuse[] = { 0.5, 0.0, 0.0, 1.0 };
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
//    glEnableClientState(GL_COLOR_ARRAY);

//    glutReshapeFunc( reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutIdleFunc(idle);
    glutMainLoop();
}

int main(int argc, char *argv[])
{
    sfield<int, float> scalar_field(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    thrust::copy(scalar_field.point_data_begin(), scalar_field.point_data_end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    threshold_geometry<sfield<int, float>, threshold_between> threshold(scalar_field, threshold_between(9, 25));
    threshold();

    threshold_p = &threshold;

    initGL(argc, argv);

    return 0;
}
