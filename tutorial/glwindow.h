/*
Copyright (c) 2012, Los Alamos National Security, LLC
All rights reserved.
Copyright 2012. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL),
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

Author: Christopher Sewell, csewell@lanl.gov
*/

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>

#include "piston/util/quaternion.h"


//===========================================================================
/*!    
    \class      GLWindow

    \brief      
    Provides callback functions for the QT OpenGL graphics window
*/
//===========================================================================
class GLWindow : public QGLWidget
{
    Q_OBJECT

public:
    //! Constructor
    GLWindow(QWidget *parent = 0);
    //! Destructor
    ~GLWindow() {};

    //! Return minimum desired window size
    QSize minimumSizeHint() const { return QSize(100, 100); };
    //! Return desired window size
    QSize sizeHint() const { return QSize(1024, 1024); };

public slots:

signals:

protected:
    //! Initialization
    void initializeGL();
    //! Rendering
    void paintGL();
    //! Handle window resize event
    void resizeGL(int width, int height);
    //! Handle mouse press event
    void mousePressEvent(QMouseEvent *event);
    //! Handle mouse move event
    void mouseMoveEvent(QMouseEvent *event);
    //! Handle key press event
    void keyPressEvent(QKeyEvent *event);

private:
    //! Mouse position at previous mouse callback event
    QPoint lastPos;
    //! QT callback timer
    QTimer *timer;
    //! OpenGL rotation matrix
    float rotationMatrix[16];
    //! Quaternion for view rotation
    Quaternion qrot;
};

#endif


