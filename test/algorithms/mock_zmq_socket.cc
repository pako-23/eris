#include "mock_zmq_socket.h"
#include "util/networking.h"
#include "zmq.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

MockZMQSocket::MockZMQSocket(int type)
    : endpoint_{}, connected_address_{}, type_{type}, subscribed_{false},
      timeout_{500}, mu_{}, cv_{}, received_{}, received_promises_{}, sent_{} {}

bool MockZMQSocket::bind(const std::string &address) noexcept {
  endpoint_ = address;
  return true;
}

bool MockZMQSocket::connect(const std::string &address) noexcept {
  connected_address_ = address;
  return true;
}

bool MockZMQSocket::send_msg(zmq_msg_t *msg, int flag) {
  std::lock_guard lk(mu_);

  sent_.emplace(*msg);
  cv_.notify_one();
  return true;
}

bool MockZMQSocket::recv_msg(zmq_msg_t *msg, int flag) {
  {
    std::lock_guard lk(mu_);

    if (!received_.empty()) {
      *msg = std::move(received_.front());
      received_promises_.front().set_value();
      received_.pop();
      received_promises_.pop();
      return true;
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(timeout_));

  {
    std::lock_guard lk(mu_);

    if (!received_.empty()) {
      *msg = std::move(received_.front());
      received_promises_.front().set_value();
      received_.pop();
      received_promises_.pop();
      return true;
    }
  }

  return false;
}

bool MockZMQSocket::setsockopt(int option, const void *optval,
                               const size_t optvallen) {
  if (option == ZMQ_RCVTIMEO) {
    memcpy(&timeout_, optval, optvallen);
    return true;
  } else if (option == ZMQ_SUBSCRIBE) {
    subscribed_ = true;
    return true;
  }

  return false;
}

std::future<void> MockZMQSocket::recv_enqueue(zmq_msg_t msg) {
  std::lock_guard lk(mu_);

  received_.emplace(msg);
  received_promises_.emplace();
  return received_promises_.back().get_future();
}

bool MockZMQSocket::send_dequeue(zmq_msg_t *msg,
                                 std::optional<unsigned> timeout) {
  std::unique_lock lk(mu_);

  if (sent_.empty()) {
    if (timeout.has_value())
      cv_.wait_for(lk, std::chrono::milliseconds(timeout.value()));
    else
      cv_.wait(lk);
  }

  if (sent_.empty())
    return false;

  *msg = std::move(sent_.front());
  sent_.pop();
  return true;
}

bool MockZMQSocket::is_empty(void) {
  std::lock_guard lk(mu_);

  return received_.empty() && sent_.empty();
}
