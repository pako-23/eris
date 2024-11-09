#pragma once

#include "algorithms/eris/coordinator.pb.h"
#include "util/networking.h"
#include <cstdint>
#include <string>

#define DEFAULT_ERIS_ROUTER_PORT 50051
#define DEFAULT_ERIS_PUBLISH_PORT 5555
#define DEFAULT_ERIS_SPLITS 1
#define DEFAULT_ERIS_ROUNDS 1
#define DEFAULT_ERIS_MIN_CLIENTS 3

/**
 * The ErisServiceConfig class contains all the configuration parameters
 * for building an ErisService.
 */
class ErisServiceConfig {
public:
  /**
   * It constructs an ErisServiceConfig with the default parameters.
   */
  explicit ErisServiceConfig(void) noexcept = default;

  /**
   * Deletes an instance of an ErisServiceConfig object.
   */
  ~ErisServiceConfig(void) noexcept = default;

  /**
   * It returns the publisher socket configuration.
   *
   * @return The publisher socket configuration.
   */
  inline const SocketConfig &get_publisher(void) const noexcept {
    return publish_address_;
  }

  /**
   * It sets the IPv4 address on which the ErisService should publish
   * updates.
   *
   * @param address The IPv4  listening address. The address must be a valid
   * IPv4 address.
   * @return It returns true if it manages to successfully set address;
   * otherwise it returns false. In practice, it returns false only if the
   * address is not a valid IPv4 address.
   */
  inline bool set_publish_address(const std::string &address) noexcept {
    return publish_address_.set_address(address);
  }

  /**
   * It sets the port on which the ErisService should publish updates.
   *
   * @param port The listening port.
   * @return It returns true if it manages to successfully set the listening
   * port; otherwise it returns false. In practice, it always returns true.
   */
  inline void set_publish_port(uint16_t port) noexcept {
    publish_address_.set_port(port);
  }

  /**
   * It returns the router socket configuration.
   *
   * @return The router socket configuration.
   */
  inline const SocketConfig &get_router(void) const noexcept {
    return router_address_;
  }

  /**
   * It sets the IPv4 address on which the ErisService should listen for
   * requests.
   *
   * @param address The IPv4  listening address. The address must be a valid
   * IPv4 address.
   * @return It returns true if it manages to successfully set address;
   * otherwise it returns false. In practice, it returns false only if the
   * address is not a valid IPv4 address.
   */
  inline bool set_router_address(const std::string &address) noexcept {
    return router_address_.set_address(address);
  }

  /**
   * It sets the port on which the ErisService should listen for incoming
   * requests.
   *
   * @param port The listening port.
   * @return It returns true if it manages to successfully set the listening
   * port; otherwise it returns false. In practice, it always returns true.
   */
  inline void set_router_port(uint16_t port) noexcept {
    router_address_.set_port(port);
  }

private:
  SocketConfig publish_address_; /**< The socket publishing updates */
  SocketConfig router_address_; /**< The router socket listening for requests */
};

/**
 * The ErisCoordinatorConfig class contains all the configuration parameters
 * for building an ErisCoordinator service.
 */
class ErisCoordinatorConfig final : public ErisServiceConfig {
public:
  /**
   * It constructs an ErisCoordinatorConfig with the default parameters.
   */
  explicit ErisCoordinatorConfig(void) noexcept;
  /**
   * Deletes an instance of an ErisCoordinatorConfig object.
   */
  ~ErisCoordinatorConfig(void) noexcept = default;

  /**
   * It sets the number of training rounds.
   *
   * @param rounds The number of training rounds. It must be a positive number.
   * @return It returns true if it manages to successfully set the number of
   * training rounds; otherwise it returns false. In practice, it returns
   * false if rounds = 0.
   */
  bool set_rounds(uint32_t rounds) noexcept;

  /**
   * It sets the number of fragments that the splitting function should produce.
   *
   * @param splits The number of fragments that the splitting function should
   * produces. It must be a positive number.
   * @return It returns true if it manages to successfully set the number of
   * fragments the splitting function should produce; otherwise it returns
   * false. In practice, it returns false if min_clients = 0.
   */
  bool set_splits(uint32_t splits) noexcept;

  /**
   * It sets the minimum number of clients that should contribute with their
   * local weights before the aggregators can publish a new model weight update.
   *
   * @param min_clients The minimum number of contributing clients. It must be a
   * positive number.
   * @return It returns true if it manages to successfully set the minimum
   * number of contributing clients; otherwise it returns false. In practice,
   * it returns false if min_clients = 0.
   */
  bool set_min_clients(uint32_t min_clients) noexcept;

  /**
   * It sets the seed used in the splitting of the model weights.
   *
   * @param split_seed The seed that should be used in the splitting of the
   * model weights.
   */
  inline void set_split_seed(uint32_t split_seed) noexcept {
    options_.set_split_seed(split_seed);
  }

  /**
   * It returns the configurations that should be used during the training.
   *
   * @return The training configurations.
   */
  inline const eris::TrainingOptions &get_options(void) const noexcept {
    return options_;
  }

private:
  eris::TrainingOptions options_; /**< The training configurations */
};

/**
 * The ErisAggregatorConfig class contains all the configuration parameters
 * for building an ErisAggregator.
 */
class ErisAggregatorConfig final : public ErisServiceConfig {
public:
  /**
   * It constructs an ErisAggregatorConfig with the given parameters.
   *
   * @param fragment_id The identifier of the model fragment as provided by the
   * ErisCoordinator.
   * @param fragment_size The size of the model fragment.
   */
  explicit ErisAggregatorConfig(uint32_t fragment_id, size_t fragment_size);

  /**
   * Deletes an instance of an ErisAggregatorConfig object.
   */
  ~ErisAggregatorConfig(void) = default;

  /**
   * It sets the minimum number of clients that should contribute with their
   * local weights before the ErisAggregator can publish a new model weight
   * update.
   *
   * @param min_clients The minimum number of contributing clients. It must be a
   * positive number.
   * @return It returns true if it manages to successfully set the minimum
   * number of contributing clients; otherwise it returns false. In practice,
   * it returns false if min_clients = 0.
   */
  bool set_min_clients(uint32_t min_clients) noexcept;

  /**
   * It returns the minimum number of clients that should contribute with their
   * local weights before the ErisAggregator can publish a new model weight
   * update.
   *
   * @return The minimum number of clients that should contribute with their
   * local weights before the ErisAggregator can publish a new model weight
   * update.
   */
  inline uint32_t get_min_client(void) const noexcept { return min_clients_; }

  /**
   * It returns the identifier of the model fragment the ErisAggregator is
   * responsible for.
   *
   * @return The identifier of the model fragment the ErisAggregator is
   * responsible for.
   */
  inline uint32_t get_fragment_id(void) const noexcept { return fragment_id_; }

  /**
   * It returns the size of the model fragment the ErisAggregator is
   * responsible for.
   *
   * @return The size of the model fragment the ErisAggregator is
   * responsible for.
   */
  inline size_t get_fragment_size(void) const noexcept {
    return fragment_size_;
  }

private:
  uint32_t fragment_id_; /**< The identifier of the fragment */
  size_t fragment_size_; /**< The size of the fragment */
  uint32_t min_clients_; /**< The number of contributing clients needed before
                            publishing a new weight update */
};
