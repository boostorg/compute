//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_RANDOM_MERSENNE_TWISTER_HPP
#define BOOST_COMPUTE_RANDOM_MERSENNE_TWISTER_HPP

#include <boost/compute/types.hpp>
#include <boost/compute/buffer.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/program.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/detail/iterator_range_size.hpp>

namespace boost {
namespace compute {

template<class T>
class mersenne_twister_engine
{
public:
    typedef T result_type;
    static const T default_seed = 5489U;
    static const T n = 624;
    static const T m = 397;

    mersenne_twister_engine(const context &context)
        : m_context(context),
          m_state_buffer(context, n * sizeof(result_type))
    {
        // setup program
        load_program();

        // seed state
        seed();
    }

    mersenne_twister_engine(const mersenne_twister_engine<T> &other)
        : m_context(other.m_context),
          m_state_buffer(other.m_state_buffer)
    {
    }

    mersenne_twister_engine<T>& operator=(const mersenne_twister_engine<T> &other)
    {
        if(this != &other){
            m_context = other.m_context;
        }

        return *this;
    }

    ~mersenne_twister_engine()
    {
    }

    void seed(result_type value = default_seed)
    {
        kernel seed_kernel = m_program.create_kernel("seed");
        seed_kernel.set_arg(0, value);
        seed_kernel.set_arg(1, m_state_buffer);

        command_queue queue(m_context, m_context.get_device());
        queue.enqueue_task(seed_kernel);
    }

    template<class OutputIterator>
    void fill(OutputIterator first, OutputIterator last, command_queue &queue)
    {
        const buffer &buffer = first.get_buffer();
        const size_t size = detail::iterator_range_size(first, last);

        kernel fill_kernel(m_program, "fill");
        fill_kernel.set_arg(0, m_state_buffer);
        fill_kernel.set_arg(1, buffer);

        size_t p = 0;

        for(;;){
            size_t count = 0;
            if(size - p >= n)
                count = n;
            else
                count = size - p;

            fill_kernel.set_arg(2, static_cast<uint_>(p));
            queue.enqueue_1d_range_kernel(fill_kernel, 0, count, 0);

            p += n;

            if(p >= size)
                break;

            generate_state(queue);
        }
    }

private:
    void generate_state(command_queue &queue)
    {
        kernel generate_state_kernel =
            m_program.create_kernel("generate_state");
        generate_state_kernel.set_arg(0, m_state_buffer);
        queue.enqueue_task(generate_state_kernel);
    }

    void load_program()
    {
        const char source[] =
            "uint twiddle(uint u, uint v)\n"
            "{\n"
            "    return (((u & 0x80000000U) | (v & 0x7FFFFFFFU)) >> 1) ^\n"
            "           ((v & 1U) ? 0x9908B0DFU : 0x0U);\n"
            "}\n"

            "__kernel void generate_state(__global uint *state)\n"
            "{\n"
            "    const uint n = 624;\n"
            "    const uint m = 397;\n"
            "    for(int i = 0; i < (n - m); i++)\n"
            "        state[i] = state[i+m] ^ twiddle(state[i], state[i+1]);\n"
            "    for(int i = n - m; i < (n - 1); i++)\n"
            "        state[i] = state[i+m-n] ^ twiddle(state[i], state[i+1]);\n"
            "    state[n-1] = state[m-1] ^ twiddle(state[n-1], state[0]);\n"
            "}\n"

            "__kernel void seed(const uint s, __global uint *state)\n"
            "{\n"
            "    const uint n = 624;\n"
            "    const uint m = 397;\n"
            "    state[0] = s & 0xFFFFFFFFU;\n"
            "    for(uint i = 1; i < n; i++){\n"
            "        state[i] = 1812433253U * (state[i-1] ^ (state[i-1] >> 30)) + i;\n"
            "        state[i] &= 0xFFFFFFFFU;\n"
            "    }\n"
            "    generate_state(state);\n"
            "}\n"

            "uint random_number(__global uint *state, const uint p)\n"
            "{\n"
            "    uint x = state[p];\n"
            "    x ^= (x >> 11);\n"
            "    x ^= (x << 7) & 0x9D2C5680U;\n"
            "    x ^= (x << 15) & 0xEFC60000U;\n"
            "    return x ^ (x >> 18);\n"
            "}\n"

            "__kernel void fill(__global uint *state,\n"
            "                   __global uint *vector,\n"
            "                   const uint offset)\n"
            "{\n"
            "    const uint i = get_global_id(0);\n"
            "    vector[offset+i] = random_number(state, i);\n"
            "}\n";

        m_program = program::create_with_source(source, m_context);
        m_program.build();
    }

private:
    context m_context;
    program m_program;
    buffer m_state_buffer;
};

typedef mersenne_twister_engine<uint_> mt19937;

} // end compute namespace
} // end boost namespace

#endif // BOOST_COMPUTE_RANDOM_MERSENNE_TWISTER_HPP
