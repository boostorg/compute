#include <iostream>
using namespace std;

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

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
    __kernel void calcVeclocity(__global const float* r, __global float* v, float dt, uint N)
    {
        const uint gid = get_global_id(0);
        const uint offset = gid*3;

        v[offset]   = 0.0f;
        v[offset+1] = 0.0f;
        v[offset+2] = 0.0f;

        uint offset2 = 0;
        float rx = 0.0;
        float ry = 0.0;
        float rz = 0.0;
        float f  = 0.0;

        for(uint i = 0; i < N; i++) {
            if(i != gid) {
                offset2 = i*3;

                rx = r[i]-r[gid];
                ry = r[i+1]-r[gid+1];
                rz = r[i+2]-r[gid+2];
                f = sqrt(rx*rx+ry*ry+rz*rz);
                f *= f*f;
                f = dt/f;

                v[offset]   += f*rx;
                v[offset+1] += f*ry;
                v[offset+2] += f*rz;
            }
        }
    }

    __kernel void updatePosition(__global float* r, __global const float* v, float dt)
    {
        const uint gid = get_global_id(0);
        const uint offset = gid*3;

        r[offset]   += dt*v[offset];
        r[offset+1] += dt*v[offset+1];
        r[offset+2] += dt*v[offset+2];
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
    void updateParticles();

private:
    bool initialPaint;

    compute::context m_context;
    compute::command_queue m_queue;
    compute::program m_program;
    compute::buffer m_v;
    compute::opengl_buffer m_r;
    compute::kernel m_kernelVelocity;
    compute::kernel m_kernelPosition;
};

NBodyWidget::NBodyWidget(QWidget* parent)
    : QGLWidget(parent), initialPaint(true)
{
}
NBodyWidget::~NBodyWidget()
{
    GLuint vbo = m_r.get_opengl_object();
    glDeleteBuffers(1, &vbo);
}

void NBodyWidget::initializeGL()
{
    // create an OpenGL interop context for OpenCL
    m_context = compute::opengl_create_shared_context();

    // get device
    compute::device device = m_context.get_device();
    cout << "device name:\t" << device.name() << endl;
    cout << "device vendor:\t" << device.vendor() << endl;

    // create a commandqueue
    m_queue = compute::command_queue(m_context, device);

    // create program
    m_program = compute::program::create_with_source(source, m_context);
    m_program.build();

    // create kernels
    m_kernelVelocity = compute::kernel(m_program, "calcVeclocity");
    m_kernelPosition = compute::kernel(m_program, "updatePosition");

    // initial particle positions
    GLuint r_vbo;
    Point* positions = new Point[NUMBER];
    boost::random::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    boost::random::mt19937_64 gen;
    for(size_t i = 0; i < NUMBER; i++) {
        positions[i].x = dist(gen);
        positions[i].y = dist(gen);
        positions[i].z = dist(gen);
    }
    glGenBuffers(1, &r_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, r_vbo);
    glBufferData(GL_ARRAY_BUFFER, NUMBER*3*sizeof(float), positions, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    m_r = compute::opengl_buffer(m_context, r_vbo);

    // allocate memory for velocities
    m_v = boost::compute::buffer(m_context, sizeof(float)*NUMBER*3);

    delete[] positions;
}
void NBodyWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
}
void NBodyWidget::paintGL()
{
    float w = width();
    float h = height();

    if(initialPaint) {
        initialPaint = false;
    } else {
        updateParticles();
    }
    
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, m_r.get_opengl_object());
    glDrawArrays(GL_POINTS, 0, NUMBER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);

    glFlush();
}
void NBodyWidget::updateParticles()
{
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    NBodyWidget widget;
    widget.show();

    return app.exec();
}

#include "nbody.moc"
