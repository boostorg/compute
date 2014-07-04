#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/random/default_random_engine.hpp>
#include <boost/compute/algorithm/shuffle.hpp>

int main(int argc, char *argv[])
{

    boost::compute::device cl_device = boost::compute::system::default_device();
    boost::compute::context cl_context(cl_device);
    boost::compute::command_queue command_queue(
                cl_context,
                cl_device,
                boost::compute::command_queue::enable_profiling);

    boost::compute::vector<float> user_data(1024);

    for(int i=0; i<1024; i++)
        user_data[i] = i;

    std::cout<< std::endl <<"user_data Before Shuffle :" <<std::endl;
    for(int i=0; i<1024; i++)
        std::cout <<user_data[i]<<" ";

    boost::compute::default_random_engine random_engine(command_queue);

    boost::compute::shuffle(user_data.begin(), user_data.end(), random_engine);

    std::cout<< std::endl <<"user_data After Shuffle :" <<std::endl;
    for(int i=0; i<1024; i++)
        std::cout <<user_data[i]<<" ";

    return 0;
}
