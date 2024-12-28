#pragma once

#include "algorithms/eris/common.pb.h"
#include "algorithms/eris/config.h"
#include "spdlog/spdlog.h"
#include "util/networking.h"
#include "zmq.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <optional>
#include <stdexcept>

/**
 * The ErisService class implements the communication services needed by the
 * eris federated training algorithm. In particular, it starts a ZeroMQ router
 * socket to handle incoming requests, and a ZeroMQ socket to publish events.
 */
template <class Socket = ZMQSocket> class ErisService final {
public:
  /**
   * It constructs an ErisService object with the provided configurations. Upon
   * construction, the process will start listening on the provided addresses.
   *
   * @param config The configuration used to build the ErisService.
   */
  explicit ErisService(const ErisServiceConfig *config)
      : publisher_{ZMQ_PUB}, router_{ZMQ_ROUTER}, running_{false} {
    if (!publisher_.bind(config->get_publisher().get_endpoint()))
      throw std::invalid_argument{"failed to bind publisher address " +
                                  config->get_publisher().get_endpoint() +
                                  ": " + strerror(errno)};

    if (!router_.bind(config->get_router().get_endpoint()))
      throw std::invalid_argument{"failed to bind router address " +
                                  config->get_router().get_endpoint() + ": " +
                                  strerror(errno)};

    const int timeout = 100;
    router_.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
  }

  /**
   * Deletes an instance of an ErisService object.
   */
  ~ErisService(void) noexcept {
    if (running_)
      stop();
  }

  /**
   * Starts the service process. In practice, it will start handling
   * client requests and publishing events.
   *
   * @param cb A callback to handle the incoming requests.
   * @param started An optional promise which will complete once the
   * service process starts listening for connections from the clients.
   */
  void
  start(std::function<void(zmq_msg_t *, zmq_msg_t *)> cb,
        std::optional<std::promise<void>> started = std::nullopt) noexcept {
    zmq_msg_t identity;
    zmq_msg_t msg;

    running_ = true;

    spdlog::info(
        "eris service started handling requests on {} and publishing on {}",
        router_.get_endpoint(), publisher_.get_endpoint());

    if (started)
      started->set_value();

    zmq_msg_init(&identity);
    zmq_msg_init(&msg);

    while (running_) {
      if (!router_.recv_msg(&identity, 0))
        continue;

      if (!router_.recv_msg(&msg, 0)) {
        zmq_msg_close(&identity);
        zmq_msg_init(&identity);
        spdlog::error("failed to recieve client's message");
        continue;
      }

      cb(&identity, &msg);

      zmq_msg_close(&msg);
      zmq_msg_close(&identity);
      zmq_msg_init(&identity);
      zmq_msg_init(&msg);
    }
  }

  /**
   * Stops the service process. In practice, it will stop handling
   * client requests and publishing events.
   */
  void stop(void) {
    spdlog::info("gracefully shutting down the eris service");
    running_ = false;
  }

  /**
   * Sends a message over the ZeroMQ router socket to the socket with the given
   * identity.
   *
   * @param identity The identifier of  the receiver socket.
   * @param message The message to send.
   * @return It returns true if it manages to successfully send the given
   * message; otherwise it returns false.
   */
  bool route_msg(zmq_msg_t *identity,
                 const google::protobuf::Message &message) noexcept {
    return router_.send_msg(identity, ZMQ_SNDMORE) &&
           send_msg(router_, message);
  }

  /**
   * Sends a message over the ZeroMQ publisher socket.
   *
   * @param message The message to send.
   * @return It returns true if it manages to successfully send the given
   * message; otherwise it returns false.
   */
  bool publish_event(const google::protobuf::Message &message) noexcept {
    return send_msg(publisher_, message);
  }

  inline Socket &get_publisher(void) { return publisher_; }
  inline Socket &get_router(void) { return router_; }

private:
  /**
   * Sends a message over a given ZeroMQ socket.
   *
   * @param sock The socket the message should be sent over.
   * @param message The message to send.
   * @return It returns true if it manages to successfully send the given
   * message; otherwise it returns false.
   */
  bool send_msg(Socket &sock,
                const google::protobuf::Message &message) noexcept {
    zmq_msg_t msg;

    zmq_msg_init_size(&msg, message.ByteSizeLong());
    message.SerializeToArray(zmq_msg_data(&msg), message.ByteSizeLong());
    return sock.send_msg(&msg, 0);
  }

  Socket publisher_;         /**< The ZeroMQ publisher socket */
  Socket router_;            /**< The ZeroMQ router socket */
  std::atomic_bool running_; /**< Whether the process is running */
};
