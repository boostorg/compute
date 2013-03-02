//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestHistogram
#include <boost/test/unit_test.hpp>

#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/algorithm/detail/histogram.hpp>
#include <boost/compute/container/vector.hpp>

BOOST_AUTO_TEST_CASE(histogram_uchar)
{
    using boost::compute::uchar_;
    using boost::compute::uint_;

    uchar_ data[] = { 1, 2, 5, 6, 7, 8, 9, 1, 5, 2,
                      0, 9, 5, 9, 8, 3, 2, 3, 4, 1,
                      4, 6, 4, 2, 8, 5, 2, 9, 7, 1 };
    boost::compute::vector<uchar_> input(data, data + 30);

    boost::compute::vector<uint_> result(10);
    boost::compute::fill(result.begin(), result.end(), uint_(0));

    boost::compute::command_queue queue =
        boost::compute::detail::default_queue_for_iterator(input.begin());

    boost::compute::detail::histogram(input.begin(),
                                      input.end(),
                                      result.begin(),
                                      queue);
    queue.finish();
    BOOST_CHECK_EQUAL(uint_(result[0]), uint_(1));
    BOOST_CHECK_EQUAL(uint_(result[1]), uint_(4));
    BOOST_CHECK_EQUAL(uint_(result[2]), uint_(5));
    BOOST_CHECK_EQUAL(uint_(result[3]), uint_(2));
    BOOST_CHECK_EQUAL(uint_(result[4]), uint_(3));
    BOOST_CHECK_EQUAL(uint_(result[5]), uint_(4));
    BOOST_CHECK_EQUAL(uint_(result[6]), uint_(2));
    BOOST_CHECK_EQUAL(uint_(result[7]), uint_(2));
    BOOST_CHECK_EQUAL(uint_(result[8]), uint_(3));
    BOOST_CHECK_EQUAL(uint_(result[9]), uint_(4));
}
