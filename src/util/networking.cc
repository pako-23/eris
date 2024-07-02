#include "util/networking.h"
#include <cstring>
#include <regex>
#include <string>

static std::string ipv4_regex =
    "(25[0-5]|2[0-4]\\d|1\\d{1,2}|[1-9]\\d?|0)(\\.(25[0-"
    "5]|2[0-4]\\d|1\\d{1,2}|[1-9]\\d?|0)){3}";
static std::string port_regex = "([1-9]\\d{0,3}|[1-5]\\d{4}|6[0-4]\\d{3}|65[0-"
                                "4]\\d{2}|655[0-2]\\d|6553[0-5])";

static std::regex ipv4_pattern{"^" + ipv4_regex + "$"};

static std::regex aggregator_submit_pattern{"^" + ipv4_regex + ":" +
                                            port_regex + "$"};

static std::regex aggregator_publish_pattern{"^tcp://" + ipv4_regex + ":" +
                                             port_regex + "$"};

bool valid_ipv4(const std::string &address) {
  return std::regex_match(address.begin(), address.end(), ipv4_pattern);
}

bool valid_aggregator_submit(const std::string &address) {
  return std::regex_match(address.begin(), address.end(),
                          aggregator_submit_pattern) &&
         strncmp(address.c_str(), "0.0.0.0", 7) != 0;
}

bool valid_aggregator_publish(const std::string &address) {
  return std::regex_match(address.begin(), address.end(),
                          aggregator_publish_pattern) &&
         strncmp(address.c_str(), "tcp://0.0.0.0", 13) != 0;
}
