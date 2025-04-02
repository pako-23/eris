#include <cerrno>
#include <chrono>
#include <cstring>
#include <new>
#include <regex>
#include <string>
#include <thread>
#include <util/networking.h>
#include <zmq.h>

static std::string ipv4_regex =
    "(25[0-5]|2[0-4]\\d|1\\d{1,2}|[1-9]\\d?|0)(\\.(25[0-"
    "5]|2[0-4]\\d|1\\d{1,2}|[1-9]\\d?|0)){3}";
static std::string port_regex = "([1-9]\\d{0,3}|[1-5]\\d{4}|6[0-4]\\d{3}|65[0-"
                                "4]\\d{2}|655[0-2]\\d|6553[0-5])";

static std::regex ipv4_pattern{"^" + ipv4_regex + "$"};

static std::regex zmq_endpoint_pattern{"^tcp://" + ipv4_regex + ":" +
                                       port_regex + "$"};

bool valid_ipv4(const std::string &address) {
  return std::regex_match(address.begin(), address.end(), ipv4_pattern);
}

bool valid_zmq_endpoint(const std::string &address) {
  return std::regex_match(address.begin(), address.end(),
                          zmq_endpoint_pattern) &&
         strncmp(address.c_str(), "tcp://0.0.0.0", 13) != 0;
}

SocketConfig::SocketConfig(void) noexcept : address_{"0.0.0.0"}, port_{0} {}

bool SocketConfig::set_address(const std::string &address) noexcept {
  if (!valid_ipv4(address))
    return false;
  address_ = address;
  return true;
}

ZMQSocket::ZMQSocket(int type) : type_{type} {
  ctx_ = zmq_ctx_new();
  if (!ctx_)
    throw std::bad_alloc{};

  socket_ = zmq_socket(ctx_, type);
  if (!socket_) {
    zmq_ctx_destroy(ctx_);
    throw std::bad_alloc{};
  }
}

ZMQSocket::~ZMQSocket(void) {
  zmq_close(socket_);
  zmq_ctx_destroy(ctx_);
}

const std::string ZMQSocket::get_endpoint(void) const noexcept {
  char end[255];
  size_t endlen = sizeof(end);

  zmq_getsockopt(socket_, ZMQ_LAST_ENDPOINT, end, &endlen);
  return std::string{end};
}

bool ZMQSocket::bind(const std::string &address) noexcept {
  if (zmq_bind(socket_, address.c_str()) != 0)
    return false;

  if (type_ == ZMQ_PUB)
    // Sleep for 200 milliseconds to prevent the slow joiner syndrome
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  return true;
}

bool ZMQSocket::connect(const std::string &address) noexcept {
  return zmq_connect(socket_, address.c_str()) == 0;
}

bool ZMQSocket::send_msg(zmq_msg_t *msg, int flag) {
  int result;

  do {
    result = zmq_msg_send(msg, socket_, flag);
  } while (result == -1 && zmq_errno() == EINTR);

  return result > 0;
}

bool ZMQSocket::recv_msg(zmq_msg_t *msg, int flag) {
  int result;

  do {
    result = zmq_msg_recv(msg, socket_, flag);
  } while (result == -1 && zmq_errno() == EINTR);

  return result >= 0;
}

bool ZMQSocket::setsockopt(int option, const void *optval,
                           const size_t optvallen) {
  return zmq_setsockopt(socket_, option, optval, optvallen);
}
