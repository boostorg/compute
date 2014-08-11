//---------------------------------------------------------------------------//
// Copyright (c) 2014 Fabian Köhler <fabian2804@googlemail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#include <iostream>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <QtGlobal>
#if QT_VERSION >= 0x050000
#include <QtWidgets>
#else
#include <QtGui>
#endif
#include <QtOpenGL>
#include <QTimer>

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute/system.hpp>
#include <boost/compute/interop/opengl.hpp>
#include <boost/compute/source.hpp>

namespace compute = boost::compute;

const uint N   = 50000;
const float dt = 0.0001f;

const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void initVelocity(__global float* velocity)
    {
        velocity[get_global_id(0)*3]   = 0.0f;
        velocity[get_global_id(0)*3+1] = 0.0f;
        velocity[get_global_id(0)*3+2] = 0.0f;
    }
    __kernel void updateVelocity(__global const float* position, __global float* velocity, float dt, uint N)
    {
        uint gid = get_global_id(0);
        uint offset_1 = gid*3;
        uint offset_2 = 0;

        float fac = 0.0f;
        float r_x = 0.0f;
        float r_y = 0.0f;
        float r_z = 0.0f;

        for(uint i = 0; i != gid; i++) {
            if(i != gid) {
                offset_2 = i*3;
                r_x = position[offset_2]-position[offset_1];
                r_y = position[offset_2+1]-position[offset_1+1];
                r_z = position[offset_2+2]-position[offset_1+2];
                fac = sqrt(r_x*r_x+r_y*r_y+r_z*r_z+0.001f);
                fac *= fac*fac;
                fac = dt/fac;
                velocity[offset_1] += fac*r_x;
                velocity[offset_1+1] += fac*r_y;
                velocity[offset_1+2] += fac*r_z;
            }
        }
    }
    __kernel void updatePosition(__global float* position, __global const float* velocity, float dt)
    {
        uint gid = get_global_id(0);
        uint offset = gid*3;

        position[offset]   += dt*velocity[offset];
        position[offset+1] += dt*velocity[offset+1];
        position[offset+2] += dt*velocity[offset+2];
    }
);

class NBodyWidget : public QGLWidget
{
    Q_OBJECT

public:
    NBodyWidget(QWidget* parent = 0);

    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void updateParticles();

private:
    QTimer* timer;

    compute::context m_context;
    compute::command_queue m_queue;
    compute::program m_program;
    compute::opengl_buffer m_position;
    compute::buffer m_velocity;
    compute::kernel m_velocity_kernel;
    compute::kernel m_position_kernel;

    bool m_initial_draw;
};

NBodyWidget::NBodyWidget(QWidget* parent)
    : QGLWidget(parent), m_initial_draw(true)
{
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
    timer->start(1);
}

void NBodyWidget::initializeGL()
{
    m_context = compute::opengl_create_shared_context();
    m_queue = compute::command_queue(m_context, m_context.get_device());
    m_program = compute::program::create_with_source(source, m_context);
    m_program.build();

    // create VBO
    float* temp = new float[N*3];
    boost::random::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    boost::random::mt19937_64 gen;
    for(size_t i = 0; i < N*3; i++) {
        temp[i] = dist(gen);
    }
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3*N*sizeof(float), temp, GL_DYNAMIC_DRAW);
    m_position = compute::opengl_buffer(m_context, vbo);

    // create buffer for velocities
    m_velocity = compute::buffer(m_context, 3*N*sizeof(float));

    // make sure velocities are 0
    compute::kernel init_kernel = m_program.create_kernel("initVelocity");
    init_kernel.set_arg(0, m_velocity);
    m_queue.enqueue_1d_range_kernel(init_kernel, 0, N, 0);
    m_queue.finish();

    // create compute kernels
    m_velocity_kernel = m_program.create_kernel("updateVelocity");
    m_velocity_kernel.set_arg(0, m_position);
    m_velocity_kernel.set_arg(1, m_velocity);
    m_velocity_kernel.set_arg(2, dt);
    m_velocity_kernel.set_arg(3, N);

    m_position_kernel = m_program.create_kernel("updatePosition");
    m_position_kernel.set_arg(0, m_position);
    m_position_kernel.set_arg(1, m_velocity);
    m_position_kernel.set_arg(2, dt);
}
void NBodyWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
}
void NBodyWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float w = width();
    float h = height();

    if(m_initial_draw) {
        m_initial_draw = false;
    } else {
        updateParticles();
    }

    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, N);

    glFinish();
}
void NBodyWidget::updateParticles()
{
    compute::opengl_enqueue_acquire_buffer(m_position, m_queue);
    m_queue.enqueue_1d_range_kernel(m_velocity_kernel, 0, N, 0).wait();
    m_queue.enqueue_1d_range_kernel(m_position_kernel, 0, N, 0).wait();
    m_queue.finish();
    compute::opengl_enqueue_release_buffer(m_position, m_queue);
}

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    NBodyWidget nbody;

    nbody.show();

    return app.exec();
}

#include "nbody.moc"
