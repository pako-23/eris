#pragma once

#include <future>
#include <optional>

/**
 * The Coordinator class provides an interface of a process that coordinates the
 * training process. In particular, it is responsible for providing
 * configurations and training parameters to the clients.
 */
class Coordinator {
public:
  /**
   * Deletes an instance of a Coordinator object.
   */
  virtual ~Coordinator(void) = default;

  /**
   * Starts the coordinator process.
   *
   * @param started An optional promise which will complete once the
   * coordinator process starts listening for connections from the clients.
   */
  virtual void
  start(std::optional<std::promise<void>> started = std::nullopt) = 0;

  /**
   * Stops the coordinator process.
   */
  virtual void stop(void) = 0;
};
