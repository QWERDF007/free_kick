#include "str_utils.h"
#include <cctype>
namespace free_kick
{
    
std::string strip(const std::string &str)
{
    auto start_it = str.begin();
    auto end_it = str.rbegin();
    while (std::isspace(*start_it))
        ++start_it;
    while (std::isspace(*end_it))
        ++end_it;
    return std::string(start_it, end_it.base());
}

} // namespace free_kick