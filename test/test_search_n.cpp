//---------------------------------------------------------------------------//
// Copyright (c) 2014 Roshan <thisisroshansmail@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestSearchN
#include <boost/test/unit_test.hpp>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/algorithm/search_n.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types/builtin.hpp>

#include "check_macros.hpp"
#include "context_setup.hpp"

namespace bc = boost::compute;

BOOST_AUTO_TEST_CASE(search_int)
{
    int data[] = {1, 2, 2, 2, 3, 2, 2, 2, 4, 6};
    bc::vector<bc::int_> vectort(data, data + 10, queue);

    bc::vector<bc::int_>::iterator iter =
        bc::search_n(vectort.begin(), vectort.end(), 2, 3, queue);

    BOOST_VERIFY(iter == vectort.begin() + 1);

    iter =
        bc::search_n(vectort.begin(), vectort.end(), 2, 5, queue);

    BOOST_VERIFY(iter == vectort.begin() + 10);
}

BOOST_AUTO_TEST_CASE(search_string)
{
    char text[] = "asaaababaaca";
    bc::vector<bc::char_> vectort(text, text + 12, queue);

    bc::vector<bc::char_>::iterator iter =
        bc::search_n(vectort.begin(), vectort.end(), 'a', 2, queue);

    BOOST_VERIFY(iter == vectort.begin() + 2);
}

BOOST_AUTO_TEST_SUITE_END()
