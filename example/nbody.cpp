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

#include <boost/program_options.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute/system.hpp>
#include <boost/compute/interop/opengl.hpp>
#include <boost/compute/source.hpp>

namespace compute = boost::compute;
namespace po = boost::program_options;

using compute::uint_;

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
                fac = sqrt(r_x*r_x+r_y*r_y+r_z*r_z+0.001f); // 0.001f is a softening factor (singularity)
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
    NBodyWidget(std::size_t particles, float dt, QWidget* parent = 0);
    ~NBodyWidget();

    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void updateParticles();
    void keyPressEvent(QKeyEvent* event);

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

    const uint_ m_particles;
    const float m_dt;
};

NBodyWidget::NBodyWidget(std::size_t particles, float dt, QWidget* parent)
    : m_initial_draw(true), m_particles(particles), m_dt(dt), QGLWidget(parent)
{
    // create a timer to redraw as fast as possible
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
    timer->start(1);
}

NBodyWidget::~NBodyWidget()
{
    // delete the opengl buffer
    GLuint vbo = m_position.get_opengl_object();
    glDeleteBuffers(1, &vbo);
}

void NBodyWidget::initializeGL()
{
    // create context, command queue and program
    m_context = compute::opengl_create_shared_context();
    m_queue = compute::command_queue(m_context, m_context.get_device());
    m_program = compute::program::create_with_source(source, m_context);
    m_program.build();

    // prepare random particle positions that will be transferred to the vbo
    float* temp = new float[m_particles*3];
    boost::random::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    boost::random::mt19937_64 gen;
    for(size_t i = 0; i < m_particles*3; i++) {
        temp[i] = dist(gen);
    }

    // create an OpenGL vbo
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3*m_particles*sizeof(float), temp, GL_DYNAMIC_DRAW);

    // create a OpenCL buffer from the vbo
    m_position = compute::opengl_buffer(m_context, vbo);
    delete[] temp;

    // create buffer for velocities
    m_velocity = compute::buffer(m_context, 3*m_particles*sizeof(float));

    // make sure velocities are 0
    compute::kernel init_kernel = m_program.create_kernel("initVelocity");
    init_kernel.set_arg(0, m_velocity);
    m_queue.enqueue_1d_range_kernel(init_kernel, 0, m_particles, 0);
    m_queue.finish();

    // create compute kernels
    m_velocity_kernel = m_program.create_kernel("updateVelocity");
    m_velocity_kernel.set_arg(0, m_position);
    m_velocity_kernel.set_arg(1, m_velocity);
    m_velocity_kernel.set_arg(2, m_dt);
    m_velocity_kernel.set_arg(3, m_particles);
    m_position_kernel = m_program.create_kernel("updatePosition");
    m_position_kernel.set_arg(0, m_position);
    m_position_kernel.set_arg(1, m_velocity);
    m_position_kernel.set_arg(2, m_dt);
}
void NBodyWidget::resizeGL(int width, int height)
{
    // update viewport
    glViewport(0, 0, width, height);
}
void NBodyWidget::paintGL()
{
    // clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // check if this is the first draw
    if(m_initial_draw) {
        // do not update particles
        m_initial_draw = false;
    } else {
        // update particles
        updateParticles();
    }

    // draw
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, m_particles);
    glFinish();
}
void NBodyWidget::updateParticles()
{
    // enqueue kernels to update particles and make sure that the command queue is finished
    compute::opengl_enqueue_acquire_buffer(m_position, m_queue);
    m_queue.enqueue_1d_range_kernel(m_velocity_kernel, 0, m_particles, 0).wait();
    m_queue.enqueue_1d_range_kernel(m_position_kernel, 0, m_particles, 0).wait();
    m_queue.finish();
    compute::opengl_enqueue_release_buffer(m_position, m_queue);
}
void NBodyWidget::keyPressEvent(QKeyEvent* event)
{
    if(event->key() == Qt::Key_Escape) {
        this->close();
    }
}

int main(int argc, char** argv)
{
    // parse command line arguments
    po::options_description options("options");
    options.add_options()
        ("help", "show usage")
        ("particles", po::value<uint_>()->default_value(1000), "number of particles")
        ("dt", po::value<float>()->default_value(0.001f), "width of each integration step");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    if(vm.count("help") > 0) {
        std::cout << options << std::endl;
        return 0;
    }

    const uint_ particles = vm["particles"].as<uint_>();
    const float dt = vm["dt"].as<float>();

    QApplication app(argc, argv);
    NBodyWidget nbody(particles, dt);

    nbody.show();

    return app.exec();
}

#include "nbody.moc"
