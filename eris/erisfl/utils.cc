
#include <regex>

static std::regex ipv4_pattern{"^\\d{1,3}(\\.\\d{1,3}){3}$"};

bool valid_ipv4(const std::string& str) {
  return std::regex_match(str.begin(), str.end(), ipv4_pattern);
}
