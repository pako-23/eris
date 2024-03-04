#pragma once

#include <cstdint>
#include <string>

bool valid_ipv4(const std::string &);
std::string zmq_listen_address(const std::string &, uint16_t);
std::string zmq_publish_address(const std::string &, uint16_t);
