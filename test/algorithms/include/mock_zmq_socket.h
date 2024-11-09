#pragma once

#include "zmq.h"
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <queue>

class MockZMQSocket {
public:
  explicit MockZMQSocket(int type);
  virtual ~MockZMQSocket(void) = default;

  virtual bool bind(const std::string &address) noexcept;
  virtual bool connect(const std::string &address) noexcept;
  inline const std::string &get_endpoint(void) const noexcept {
    return endpoint_;
  }
  inline const std::string &get_connected_address(void) const noexcept {
    return connected_address_;
  }
  bool send_msg(zmq_msg_t *msg, int flag);
  bool recv_msg(zmq_msg_t *msg, int flag);
  bool setsockopt(int option, const void *optval, const size_t optvallen);

  std::future<void> recv_enqueue(zmq_msg_t msg);
  bool send_dequeue(zmq_msg_t *msg,
                    std::optional<unsigned> timout = std::nullopt);

  bool is_empty(void);
  inline bool subscribed(void) const { return type_ == ZMQ_SUB && subscribed_; }

private:
  std::string endpoint_;
  std::string connected_address_;
  int type_;
  bool subscribed_;

  int timeout_;
  std::mutex mu_;
  std::condition_variable cv_;

  std::queue<zmq_msg_t> received_;
  std::queue<std::promise<void>> received_promises_;

  std::queue<zmq_msg_t> sent_;
};
