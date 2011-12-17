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
#include "glwindow.h"

#include <QtGui>
#include <QtOpenGL>
#include <QObject>

#include <math.h>

GLWindow::GLWindow(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    isoRender = new IsoRender();

    renderFlag = -1;
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
    timer->start(1);
}


GLWindow::~GLWindow()
{
    isoRender->cleanup();
}


QSize GLWindow::minimumSizeHint() const
{
    return QSize(100, 50);
}


QSize GLWindow::sizeHint() const
{
    return QSize(2048, 1024);
}


static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}


void GLWindow::setIsovalue(int value)
{
    isoRender->setIsovaluePct(value/100.0);
}


void GLWindow::setPlaneLevel(int value)
{
    isoRender->setPlaneLevelPct(value/100.0);
}


void GLWindow::setDataSet1(bool enabled)
{
    if (enabled) renderFlag = 1;
}


void GLWindow::setDataSet2(bool enabled)
{
    if (enabled) renderFlag = 2;
}


void GLWindow::setDataSet3(bool enabled)
{
    if (enabled) renderFlag = 3;
}


void GLWindow::setDataSet4(bool enabled)
{
    if (enabled) renderFlag = 4;
}


void GLWindow::resetView()
{
    isoRender->resetView();
}


void GLWindow::setShowIsosurface(bool show)
{
    isoRender->includeContours =  isoRender->useContours && show;
    isoRender->includeThreshold = isoRender->useThreshold && show;
    isoRender->includeConstantContours = isoRender->useConstantContours && show;
}


void GLWindow::setShowCutPlane(bool show)
{
    isoRender->includePlane = show;
}


void GLWindow::initialize(int dataSetIndex, bool aBigDemo)
{
    if (dataSetIndex >= 0) isoRender->read(dataSetIndex);
    bigDemo = aBigDemo;

    setPlaneSlider(isoRender->planeLevelPct*100.0);
    setIsoSlider(isoRender->isovaluePct*100.0);
    setShowIsosurfaceCheckBox(true);
    setShowClipPlaneCheckBox(true);
}


void GLWindow::initializeGL()
{
    isoRender->initGL(true, bigDemo, true, bigDemo ? 5 : 1);

    setPlaneSlider(isoRender->planeLevelPct*100.0);
    setIsoSlider(isoRender->isovaluePct*100.0);
    setShowIsosurfaceCheckBox(true);
    setShowClipPlaneCheckBox(true);
}


void GLWindow::paintGL()
{
    timer->stop();
    isoRender->display();
    if (renderFlag >= 0)
    {
      initialize(renderFlag, bigDemo);
      renderFlag = -1;
    }
    timer->start(1);
}


void GLWindow::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
    isoRender->viewportWidth = width;
    isoRender->viewportHeight = height;
}


void GLWindow::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}


void GLWindow::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton)
    {
      Quaternion newRotX;
      newRotX.setEulerAngles(-0.2*dx*3.14159/180.0, 0.0, 0.0);
      isoRender->qrot.mul(newRotX);

      Quaternion newRotY;
      newRotY.setEulerAngles(0.0, 0.0, -0.2*dy*3.14159/180.0);
      isoRender->qrot.mul(newRotY);
    }
    else if (event->buttons() & Qt::RightButton)
    {
      isoRender->setZoomLevelPct(isoRender->zoomLevelPct + dy/1000.0);
    }
    lastPos = event->pos();
}


