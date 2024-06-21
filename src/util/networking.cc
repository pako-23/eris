#include "util/networking.h"
#include <regex>
#include <string>

static std::string ipv4_regex =
    "(25[0-5]|2[0-4]\\d|1\\d{1,2}|[1-9]\\d?|0)(\\.(25[0-"
    "5]|2[0-4]\\d|1\\d{1,2}|[1-9]\\d?|0)){3}";
static std::string port_regex = "([1-9]\\d{0,3}|[1-5]\\d{4}|6[0-4]\\d{3}|65[0-"
                                "4]\\d{2}|655[0-2]\\d|6553[0-5])";

static std::regex ipv4_pattern{"^" + ipv4_regex + "$"};

static std::regex aggregator_pattern{"^" + ipv4_regex + ":" + port_regex + "$"};

bool valid_ipv4(const std::string &address) {
  return std::regex_match(address.begin(), address.end(), ipv4_pattern);
}

bool valid_aggregator(const std::string &aggregator) {
  return std::regex_match(aggregator.begin(), aggregator.end(),
                          aggregator_pattern);
}
