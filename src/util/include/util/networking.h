#pragma once

#include "zmq.h"
#include <cstddef>
#include <cstdint>

#include <string>

/**
 * Checks if address contains a valid IPv4 address.
 *
 * @param address A string containing the IPv4 address.
 * @return true if the address contains a valid IPv4 address; otherwise false.
 */
bool valid_ipv4(const std::string &address);

/**
 * Checks if address contains a valid ZeroMQ endpoint address.
 * A ZeroMQ endpoit address is valid if it has the form
 * tcp://<address>:<port>, where address must be a valid IPv4 address and port
 * must be a valid port number. Also, address cannot be 0.0.0.0 nor *.
 *
 * @param address A string containing a ZeroMQ endpoint address.
 * @return true If the address contains a valid ZeroMQ endpoint address;
 * otherwise false.
 */
bool valid_zmq_endpoint(const std::string &address);

/**
 * The SocketConfig class contains all the configuration parameters
 * for building a ZeroMQ socket.
 */
class SocketConfig final {
public:
  /**
   * It constructs a SocketConfig with the default parameters. With the default
   * parameters, the socket will have address 0.0.0.0 and port 0 (the listening
   * port will be dynamically allocated by the OS).
   */
  explicit SocketConfig(void) noexcept;

  /**
   * It constructs a SocketConfig from the given endpoint address.
   *
   * @param endpoint The endepoint from which the socket configuration should be
   * generated.
   */
  explicit SocketConfig(const std::string &endpoint);

  /**
   * Deletes an instance of an SocketConfig object.
   */
  ~SocketConfig(void) noexcept = default;

  /**
   * It sets the IPv4 address for the socket.
   *
   * @param address The IPv4 listening address. The address must be a valid
   * IPv4 address.
   * @return It returns true if it manages to successfully set the address;
   * otherwise it returns false. In practice, it returns false only if the
   * address is not a valid IPv4 address.
   */
  bool set_address(const std::string &address) noexcept;

  /**
   * It sets listening port for the socket.
   *
   * @param port The socket listening port. A port with value 0 indicates that
   * the port should be dynamically allocated by the OS.
   */
  inline void set_port(uint16_t port) noexcept { port_ = port; }

  /**
   * It returns the full address of the socket. The address will be of the form
   * tcp://<address>:<port> where address is the listening IPv4 address and
   * port is the listening port.
   *
   * @return The listening address of the socket.
   */
  inline std::string get_endpoint(void) const noexcept {
    std::string address = address_ == "0.0.0.0" ? "*" : address_;
    return "tcp://" + address + ":" + std::to_string(port_);
  }

private:
  std::string address_; /**< The IP address of the socket */
  uint16_t port_;       /**< The port of the socket */
};

/**
 * The ZMQSocket class is a wrapper around a ZeroMQ socket.
 */
class ZMQSocket final {
public:
  /**
   * It constructs a ZMQSocket using the provided configuration and type.
   *
   * @param config The address configuration of the socket.
   * @param type The ZeroMQ socket type.
   */
  explicit ZMQSocket(int type);

  /**
   * Deletes an instance of an ZMQSocket object. It closes the underlaying
   * ZeroMQ socket and releases its context.
   */
  ~ZMQSocket(void);

  /**
   * It returns the bind address of the socket. The address will be of the form
   * tcp://<address>:<port> where address is the listening IPv4 address and
   * port is the listening port.
   *
   * @return The listening address of the socket.
   */
  const std::string get_endpoint(void) const noexcept;

  /**
   * It binds the socket to a given address. The address has to match
   * tcp://<address>:<port> where address is the listening IPv4 address and
   * port is the listening port.
   *
   * @param The bind address of the socket.
   */
  bool bind(const std::string &address) noexcept;

  /**
   * It connects the scoket to a given address. The address has to match
   * tcp://<address>:<port> where address is the listening IPv4 address and
   * port is the listening port. The addresse 0.0.0.0 and port 0 are not
   * allowed.
   *
   * @param The addres the socket should connect to.
   */
  bool connect(const std::string &address) noexcept;

  /**
   * It sends a ZeroMQ message over the socket.
   *
   * @param msg The ZeroMQ message to be sent over the socket.
   * @param flag ZeroMQ sending flags.
   * @return It returns true if it manages to successfully send the given
   * message; otherwise it returns false.
   */
  bool send_msg(zmq_msg_t *msg, int flag);

  /**
   * It receives a ZeroMQ message from the socket.
   *
   * @param msg The ZeroMQ message in which the message should be received.
   * @param flag ZeroMQ receiving flags.
   * @return It returns true if it manages to successfully receive some message
   * from the socket; otherwise it returns false.
   */
  bool recv_msg(zmq_msg_t *msg, int flag);

  /**
   * It sets some given option on the a ZeroMQ socket.
   *
   * @param option The option to set.
   * @param flag The value of the option.
   * @param optvallen The length of the option value.
   * @return It returns true if it manages to successfully set the socket
   * option; otherwise it returns false.
   */
  bool setsockopt(int option, const void *optval, const size_t optvallen);

private:
  void *ctx_;    /**< The ZeroMQ socket context */
  void *socket_; /**< The ZeroMQ socket */
  int type_;     /**< The type of the socket */
};
