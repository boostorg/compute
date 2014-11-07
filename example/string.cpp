//---------------------------------------------------------------------------//
// Copyright (c) 2013-2014 Mageswaran.D <mageswaran1989@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include <iostream>
#include <string>
#include <boost/compute/container/string.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/program_options.hpp>

namespace compute = boost::compute;
namespace po = boost::program_options;


/***
 *  This example is about manipulating strings using
 *  boost::compute::string APIs
 *
 **/
int main(int argc, char *argv[])
{
    // setup the command line arguments
    po::options_description desc;
    desc.add_options()
            ("help",  "string example using compute::string");

    // Parse the command lines
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //check the command line arguments
    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    //Declaring the string
    compute::string bc_str;
    //Intializing the string
    compute::string bc_str1 = "boost::compute::string";
    compute::string bc_str2("3rd way of initializing");
    compute::string bc_sub_str;
    //String iterator
    compute::string::iterator iter;

    bc_str = bc_str1;

    std::cout << "bc_str           : ";
    //Accessing individual elements
    for(unsigned int i=0; i < bc_str.length(); i++)
        std::cout << bc_str[i];
    std::cout << "\n";

    //To know the length of the string
    std::cout << "size             : " << bc_str.size() << "\n";
    std::cout << "length           : " << bc_str.length() << "\n";
    std::cout << "capacity(bytes)  : " << bc_str.capacity() << "\n";

    //Accessing individual elements
    std::cout << "at(7)            : " << bc_str.at(7) << "\n";

    //Check the whether string is empty or not
    std::cout << "empty(0/1)       : " << bc_str.empty() << "\n";

    //Substring
    bc_sub_str = bc_str.substr(7,14);
    std::cout << "substring        : ";
    //Accessing individual elements
    for(unsigned int i=0; i<bc_sub_str.length(); i++)
        std::cout << bc_sub_str[i];
    std::cout << "\n";

    // (CL_DEVICE_MAX_MEM_ALLOC_SIZE / sizeof(char) )
    std::cout << "max size of string in the running device   : "
              << bc_str.max_size()/1024 << "KB\n";

    //string find
    std::size_t found_pos = bc_str.find('c');
    std::cout << "c found at position   : " << found_pos <<"\n";

    std::cout << "bc_str2          : ";
    //Accessing individual elements via iterator
    for(iter = bc_str2.begin(); iter != bc_str2.end(); iter++)
        std::cout << *iter;
    std::cout << "\n\n";

    std::cout << "Clearing bc_str" << "\n";
    //string clear
    bc_str.clear();

    std::cout << "Swaping bc_str & bc_str2" << "\n";
    //string swap
    bc_str2.swap(bc_str);
    std::cout << "After swap" << "\n";

    std::cout << "bc_str           : ";
    //Accessing individual elements
    for(unsigned int i=0; i < bc_str.length(); i++)
        std::cout << bc_str[i];
    std::cout << "\n";

    std::cout << "bc_str2          : ";
    //Accessing individual elements
    for(unsigned int i=0; i < bc_str2.length(); i++)
        std::cout << bc_str2[i];
    std::cout << "\n";

    return 0;
}
