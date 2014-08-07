#include <iostream>
using namespace std;

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

#include <QtGlobal>
#if QT_VERSION >= 0x050000
#include <QtWidgets>
#else
#include <QtGui>
#endif
#include <QtOpenGL>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/source.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/interop/opengl.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#define NUMBER 10

namespace compute = boost::compute;

struct Point
{
    GLfloat x, y, z;
};

const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void calcVeclocity()
    {
    }

    __kernel void updatePosition()
    {
    }
);

class NBodyWidget : public QGLWidget
{
    Q_OBJECT

public:
        NBodyWidget(QWidget* parent = 0);
        ~NBodyWidget();

        void initializeGL();
        void resizeGL(int width, int height);
        void paintGL();

private:
        compute::context m_context;
        compute::command_queue m_queue;
        compute::program m_program;
        compute::buffer m_v;

        GLuint m_r;
};

NBodyWidget::NBodyWidget(QWidget* parent)
    : QGLWidget(parent)
{
}
NBodyWidget::~NBodyWidget()
{
    glDeleteBuffers(1, &m_r);
}

void NBodyWidget::initializeGL()
{
    // create an OpenGL interop context for OpenCL
    m_context = compute::opengl_create_shared_context();

    // get device
    compute::device device = m_context.get_device();
    cout << "device name:\t" << device.name() << endl;
    cout << "device vendor:\t" << device.vendor() << endl;

    // crearte a commandqueue
    m_queue = compute::command_queue(m_context, device);

    // initial particle positions
    Point* positions = new Point[NUMBER];
    boost::random::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    boost::random::mt19937_64 gen;
    for(size_t i = 0; i < NUMBER; i++) {
        positions[i].x = dist(gen);
        positions[i].y = dist(gen);
        positions[i].z = 0.0f;
    }
    glGenBuffers(1, &m_r);
    glBindBuffer(GL_ARRAY_BUFFER, m_r);
    glBufferData(GL_ARRAY_BUFFER, NUMBER*3*sizeof(float), positions, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // allocate memory for velocities
    m_v = boost::compute::buffer(m_context, sizeof(float)*NUMBER*2);

}
void NBodyWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
}
void NBodyWidget::paintGL()
{
    float w = width();
    float h = height();
    
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, m_r);
    glDrawArrays(GL_POINTS, 0, NUMBER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);

    glFlush();
}
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    NBodyWidget widget;
    widget.show();

    return app.exec();
}

#include "nbody.moc"
