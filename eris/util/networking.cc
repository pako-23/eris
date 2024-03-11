#include "util/networking.h"
#include <arpa/inet.h>
#include <cstring>
#include <fstream>
#include <functional>
#include <ios>
#include <net/if.h>
#include <netinet/in.h>
#include <regex>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

static std::regex ipv4_pattern{"^\\d{1,3}(\\.\\d{1,3}){3}$"};

bool valid_ipv4(const std::string &str) {
  return std::regex_match(str.begin(), str.end(), ipv4_pattern);
}

static std::string
zmq_address(const std::string &address, uint16_t port,
            const std::function<std::string(const std::string &)> &transform) {
  std::ostringstream ss;
  ss << "tcp://" << transform(address) << ':' << port;
  return ss.str();
}

static std::string get_default_interface(void) {
  std::ifstream routes("/proc/net/route", std::ios_base::in);

  if (!routes.good())
    return "";

  std::string line;
  std::vector<std::string> tokens;

  std::string interface, destination;

  while (std::getline(routes, line)) {
    std::istringstream ss(line);

    ss >> interface >> destination;
    if (destination == "00000000") {
      routes.close();
      return interface;
    }
  }
  routes.close();
  return "";
}

static std::string get_interface_address(const std::string &interface) {
  char buf[INET_ADDRSTRLEN];
  struct ifreq req;
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    return "";

  req.ifr_addr.sa_family = AF_INET;
  std::strncpy(req.ifr_name, interface.c_str(), IFNAMSIZ - 1);
  if (ioctl(fd, SIOCGIFADDR, &req) < 0) {
    close(fd);
    return "";
  }
  close(fd);

  if (!inet_ntop(AF_INET, &((struct sockaddr_in *)&req.ifr_addr)->sin_addr, buf,
                 INET_ADDRSTRLEN))
    return "";
  return std::string{buf};
}

std::string zmq_listen_address(const std::string &address, uint16_t port) {
  return zmq_address(address, port, [](const std::string &address) {
    return address == "0.0.0.0" ? "*" : address;
  });
}

std::string zmq_publish_address(const std::string &address, uint16_t port) {
  return zmq_address(address, port, [](const std::string &address) {
    if (address == "0.0.0.0")
      return get_interface_address(get_default_interface());
    return address;
  });
}
