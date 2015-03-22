//---------------------------------------------------------------------------//
// Copyright (c) 2013 Muhammad Junaid Muzammil <mjunaidmuzammil@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//


#include <boost/compute/random/threefry.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/system.hpp>
#include <iostream>

int main() 
{
    using boost::compute::uint_;
    boost::compute::device device = boost::compute::system::default_device();
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);
    boost::compute::random::threefry rng(queue);
    boost::compute::vector<uint_> vector_ctr(20, context);
    boost::compute::vector<uint_> vector_key(20, context);
    
    uint32_t ctr[20];
    uint32_t key[20];
    for(int i = 0; i < 10; i++) {
        ctr[i*2] = i;
        ctr[i*2+1] = 0;
        key[i*2] = 0;
        key[i*2+1] = 0; 
    }
    boost::compute::copy(ctr, ctr+20, vector_ctr.begin(), queue);
    boost::compute::copy(key, key+20, vector_key.begin(), queue);
    rng.generate(queue, vector_ctr.begin(), vector_ctr.end(), vector_key.begin(), vector_key.end());
    boost::compute::copy(vector_ctr.begin(), vector_ctr.end(), ctr, queue);

    for(int i = 0; i < 10; i++) {
        std::cout << std::hex << ctr[i*2] << " " << ctr[i*2+1] << std::endl;
    }
    return 0;
}

