// Copyright 2024 Peter Dimov
// Distributed under the Boost Software License, Version 1.0
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/compute/detail/sha1.hpp>
#include <boost/core/lightweight_test.hpp>

std::string digest( std::string const& s )
{
    boost::compute::detail::sha1 h;
    h.process( s );
    return h;
}

int main()
{
    // https://en.wikipedia.org/wiki/SHA-1#Example_hashes

    BOOST_TEST_EQ(
        digest( "The quick brown fox jumps over the lazy dog" ),
        std::string( "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12" ) );

    BOOST_TEST_EQ(
        digest( "The quick brown fox jumps over the lazy cog" ),
        std::string( "de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3" ) );

    BOOST_TEST_EQ(
        digest( "" ),
        std::string( "da39a3ee5e6b4b0d3255bfef95601890afd80709" ) );

    return boost::report_errors();
}
