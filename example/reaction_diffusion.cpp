//---------------------------------------------------------------------------//
// Copyright (c) 2020 Adam Wulkiewicz
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#include <boost/compute.hpp>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

namespace bc = boost::compute;

// Inspired by Reaction-Diffusion Tutorial by Karl Sims 
// see: http://www.karlsims.com/rd.html
class reaction
{
public:
    reaction(size_t w, size_t h)
        : m_device(bc::system::default_device())
        , m_context(m_device)
        , m_queue(m_context, m_device)
        , m_program(m_context)
        , m_kernel(m_program.program, "my_program")
        , m_current(0)
        , m_w(w)
        , m_h(h)
    {
        std::vector<bc::float2_> vec(w * h);
        for (size_t j = 0; j < h; j++)
        {
            for (size_t i = 0; i < w; i++)
            {
                size_t id = j * w + i;
                vec[id].x = 1;
                vec[id].y = 0;
            }
        }

        for (size_t j = h / 2 - 5; j < h / 2 + 5; j++)
        {
            for (size_t i = w / 2 - 5; i < w / 2 + 5; i++)
            {
                size_t id = j * w + i;
                vec[id].y = 1;
            }
        }

        m_device_vectors[0] = bc::vector<bc::float2_>(w * h, m_context);
        m_device_vectors[1] = bc::vector<bc::float2_>(w * h, m_context);
        m_current = 0;

        bc::copy(vec.begin(), vec.end(), m_device_vectors[0].begin(), m_queue);
        bc::copy(m_device_vectors[0].begin(), m_device_vectors[0].end(), m_device_vectors[1].begin(), m_queue);
    }

    template <typename Vector>
    void next(unsigned n, Vector & texture)
    {
        BOOST_ASSERT(texture.size() == m_w * m_h);

        for (unsigned i = 0 ; i < n ; i++)
            calculate_next();

        bc::vector<bc::float2_> const& current_vector = m_device_vectors[m_current];
        std::vector<bc::float2_> vec(m_w * m_h);
        bc::copy(current_vector.begin(), current_vector.end(), vec.begin(), m_queue);
        for (size_t k = 0; k < vec.size(); k++)
            texture[k] = luminosity(vec[k]);
    }

private:
    void calculate_next()
    {
        size_t next = (m_current + 1) % 2;
        bc::vector<bc::float2_> const& current_vector = m_device_vectors[m_current];
        bc::vector<bc::float2_> const& next_vector = m_device_vectors[next];
        m_current = next;

        m_kernel.set_arg(0, current_vector.get_buffer());
        m_kernel.set_arg(1, next_vector.get_buffer());
        m_kernel.set_arg(2, (bc::uint_)m_w);
        m_kernel.set_arg(3, (bc::uint_)m_h);

        size_t origin[2] = { 1, 1 };
        size_t region[2] = { m_w - 2, m_h - 2 };

        m_queue.enqueue_nd_range_kernel(m_kernel, 2, origin, region, 0);
        m_queue.finish();
    }

    static unsigned char luminosity(bc::float2_ const& f2)
    {
        //float val = bounded((f2.x - f2.y - 0.5f) * 2.0f, -0.5f, 0.5f) + 0.5;
        float val = bounded((f2.x - f2.y) * 2.0f, 0.0f, 1.0f);
        return (unsigned char)(255 * val);
    }

    template <typename T>
    static T bounded(T val, T mi, T ma)
    {
        return (std::min)((std::max)(val, mi), ma);
    }

    struct program_holder
    {
        program_holder(bc::context const& context)
            : program(bc::program::create_with_source(source(), context))
        {
            program.build();
        }

        static const char * source()
        {
            return BOOST_COMPUTE_STRINGIZE_SOURCE(
                __kernel void my_program(__global __read_only float2* curr,
                                         __global __write_only float2* next,
                                         uint w,
                                         uint h)
                {
                    uint i = get_global_id(0);
                    uint j = get_global_id(1);
                
                    // Parameters
                    float da = 1.0f;
                    float db = 0.5f;
                    //float f = 0.04f;
                    //float k = 0.0649f;
                    //float f = 0.0545f;
                    //float k = 0.062f;
                    //float f = 0.055f;
                    //float k = 0.062f;
                    float f = (float)j / h * (0.07f - 0.03f) + 0.01f;
                    float k = (float)i / w * (0.07f - 0.045f) + 0.045f;

                    // 2D Laplacian
                    // id_0 - 1 | id_0 | id_0 + 1
                    // --------------------------
                    // id_1 - 1 | id_1 | id_1 + 1
                    // --------------------------
                    // id_2 - 1 | id_2 | id_2 + 1
                    int id_0 = (j - 1) * w + i;
                    int id_1 = j * w + i;
                    int id_2 = (j + 1) * w + i;
                    float2 l = 0.05f * curr[id_0 - 1]
                        + 0.2f * curr[id_0]
                        + 0.05f * curr[id_0 + 1]
                        + 0.2f * curr[id_1 - 1]
                        - 1 * curr[id_1]
                        + 0.2f * curr[id_1 + 1]
                        + 0.05f * curr[id_2 - 1]
                        + 0.2f * curr[id_2]
                        + 0.05f * curr[id_2 + 1];

                    // New values
                    float a = curr[id_1].x;
                    float b = curr[id_1].y;
                    float la = l.x;
                    float lb = l.y;
                    float abb = a * b * b;
                    float na = a + (da * la - abb + f * (1 - a));
                    float nb = b + (db * lb + abb - (k + f) * b);

                    // Normaliziation
                    if (na < 0.0f) na = 0.0f;
                    else if (na > 1.0f) na = 1.0f;
                    if (nb < 0.0f) nb = 0.0f;
                    else if (nb > 1.0f) nb = 1.0f;

                    next[id_1].x = na;
                    next[id_1].y = nb;
                }
            );
        }

        bc::program program;
    };

    bc::device m_device;
    bc::context m_context;
    bc::command_queue m_queue;
    program_holder m_program;
    bc::kernel m_kernel;

    bc::vector<bc::float2_> m_device_vectors[2];
    size_t m_current;

    size_t m_w;
    size_t m_h;
};

int width = 800;
int height = 800;

reaction react(width, height);

void init()
{
    GLuint texture_id;
    glEnable(GL_TEXTURE_2D);    
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

void render_scene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex3f(-1, -1, 0);
        glTexCoord2f(1, 0);
        glVertex3f(1, -1, 0);
        glTexCoord2f(1, 1);
        glVertex3f(1, 1, 0);
        glTexCoord2f(0, 1);
        glVertex3f(-1, 1, 0);
    glEnd();

    glutSwapBuffers();
}

void idle()
{
    std::vector<unsigned char> texture(width * height);
    react.next(64, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, &texture[0]);

    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(width, height);
    glutCreateWindow("Boost.Compute - Reaction Diffusion");
    init();
    glutDisplayFunc(render_scene);
    glutIdleFunc(idle);
    glutMainLoop();
}
