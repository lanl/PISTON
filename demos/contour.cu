/*
Copyright (c) 2011, Los Alamos National Security, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
    	and/or other materials provided with the distribution.
    Neither the name of the Los Alamos National Laboratory nor the names of its contributors may be used to endorse or promote products derived from this
    	software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "isorender.h"

#include <GL/glut.h>

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;

IsoRender* isorender;


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

    if (mouse_buttons == 1)
    {
      Quaternion newRotX;
      newRotX.FromEuler(-0.2*dx, 0.0, 0.0);
      isorender->qrot = isorender->qrot * newRotX;

      Quaternion newRotY;
      newRotY.FromEuler(0.0, 0.0, -0.2*dy);
      isorender->qrot = isorender->qrot * newRotY;
    }
    else if (mouse_buttons == 2)
    {
      isorender->setZoomLevelPct(isorender->zoomLevelPct + dy/1000.0);
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}


void keyboard( unsigned char key, int x, int y )
{
    switch (key)
    {
      case 'q': isorender->isovalue += 10.0; break;
      case 'w': isorender->isovalue -= 10.0; break;
      case 'a': isorender->animate = !(isorender->animate);
    }
}


void display()
{
    isorender->display();
    glutSwapBuffers();
}


void idle()
{
    glutPostRedisplay();
}


void initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(2048, 1024);
    glutCreateWindow("Marching Cube");

    isorender->initGL(true, false, false, atoi(argv[1]));

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutIdleFunc(idle);
    glutMainLoop();
}


int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: contour[GPU/OMP] dataSetIndex numIters" << std::endl;
		return 0;
	}
    int numIters = 0;
    if (argc > 2) numIters = atoi(argv[2]);

    isorender = new IsoRender();
    
    if (numIters <= 0)
    {
    	initGL(argc, argv);
    }
    else
    {
    	glutInit(&argc, argv);
    	isorender->initGL(false, false, false, atoi(argv[1]));
    	isorender->numIters = numIters;
    	isorender->timeContours();
    }

    return 0;
}

