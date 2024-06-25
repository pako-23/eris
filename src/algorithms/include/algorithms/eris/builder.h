#pragma once

#include "algorithms/eris/coordinator.pb.h"
#include <cstddef>
#include <cstdint>
#include <string>

#define DEFAULT_ERIS_RPC_PORT 50051
#define DEFAULT_ERIS_PUBSUB_PORT 5555
#define DEFAULT_ERIS_SPLITS 1
#define DEFAULT_ERIS_ROUNDS 1
#define DEFAULT_ERIS_MIN_CLIENTS 3

using eris::TrainingOptions;

/**
 * The ErisServiceBuilder is a base class for building an eris service
 * builder. A builder will carry configuration parameters for building an
 * eris service. The ErisCoordinator is an example fo an eris service.
 */
class ErisServiceBuilder {
protected:
  /**
   * It constructs a builder with the default parameters.
   */
  explicit ErisServiceBuilder(void) noexcept;

public:
  /**
   * Deletes an instance of an ErisServiceBuilder object.
   */
  virtual ~ErisServiceBuilder(void) noexcept = default;

  /**
   * It sets the port on which the service should listen for incoming RPC
   * requests.
   *
   * @param port The RPC listening port.
   * @return It returns true if it manages to successfully set the RPC port;
   * otherwise it returns false. In practice, it always returns true.
   */
  bool add_rpc_port(uint16_t port) noexcept;

  /**
   * It sets the IPv4 address on which the service should listen for incoming
   * RPC requests.
   *
   * @param address The IPv4 RPC listening address. The address must be a valid
   * IPv4 address.
   * @return It returns true if it manages to successfully set the RPC address;
   * otherwise it returns false. In practice, it returns false only if the
   * address is not a valid IPv4 address.
   */
  bool add_rpc_listen_address(const std::string &address) noexcept;

  /**
   * It sets the port on which the service should publish events.
   *
   * @param port The Pub-Sub publishing port.
   * @return It returns true if it manages to successfully set the Pub-Sub port;
   * otherwise it returns false. In practice, it always returns true.
   */
  bool add_publish_port(uint16_t port) noexcept;

  /**
   * It sets the IPv4 address on which the service should publish events.
   *
   * @param address The IPv4 publishing address. The address must be a valid
   * IPv4 address.
   * @return It returns true if it manages to successfully set the publishing
   * address; otherwise it returns false. In practice, it returns false only if
   * the address is not a valid IPv4 address.
   */
  bool add_publish_address(const std::string &address) noexcept;

  /**
   * It returns the full address on which the service should listen for RPC
   * requests. The address will be of the form <address>:<port> where address is
   * the RPC listening IPv4 address and port is the RPC listening port.
   *
   * @return The listening address on which the service should listen for RPC
   * requests
   */
  inline const std::string get_rpc_listen_address(void) const noexcept {
    return rpc_listen_address_ + ":" + std::to_string(rpc_port_);
  }

  /**
   * It returns the full address on which the service should publish events.
   * The address will be a valid ZeroMQ DSN, so it will have the form
   * tcp://<address>:<port> where address is the publisher IPv4 address and port
   * is the publisher port.
   *
   * @return The ZeroMQ DSN identifying the publishing address.
   */
  inline const std::string get_pubsub_listen_address(void) const noexcept {
    std::string address =
        publish_address_ == "0.0.0.0" ? "*" : publish_address_;
    return "tcp://" + address + ":" + std::to_string(publish_port_);
  }

protected:
  std::string rpc_listen_address_; /**< The IPv4 address of the RPC server */
  uint16_t rpc_port_;              /**< The port of the RPC server */

  std::string publish_address_; /**< The IPv4 address of the publisher */
  uint16_t publish_port_;       /**< The port of the publisher */
};

/**
 * The ErisCoordinatorBuilder class contains all the configuration parameters
 * for building an ErisCoordinator.
 */
class ErisCoordinatorBuilder final : public ErisServiceBuilder {
public:
  /**
   * It constructs a builder with the default parameters.
   */
  explicit ErisCoordinatorBuilder(void) noexcept;

  /**
   * Deletes an instance of an ErisCoordinatorBuilder object.
   */
  virtual ~ErisCoordinatorBuilder(void) noexcept = default;

  /**
   * It sets the number of training rounds.
   *
   * @param rounds The number of training rounds. It must be a positive number.
   * @return It returns true if it manages to successfully set the number of
   * training rounds; otherwise it returns false. In practice, it returns
   * false if rounds = 0.
   */
  bool add_rounds(uint32_t rounds) noexcept;

  /**
   * It sets the number of fragments that the splitting function should produce.
   *
   * @param splits The number of fragments that the splitting function should
   * produces. It must be a positive number.
   * @return It returns true if it manages to successfully set the number of
   * fragments the splitting function should produce; otherwise it returns
   * false. In practice, it returns false if min_clients = 0.
   */
  bool add_splits(uint32_t splits) noexcept;

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
  bool add_min_clients(uint32_t min_clients) noexcept;

  /**
   * It sets the seed used in the splitting of the model weights.
   *
   * @param split_seed The seed that should be used in the splitting of the
   * model weights.
   * @return It returns true if it manages to successfully set the splitting
   * seed; otherwise it returns false. In practice, it always returns true.
   */
  bool add_split_seed(uint32_t split_seed) noexcept;

  /**
   * It returns the configurations that should be used during the training.
   *
   * @return The training configurations.
   */
  inline const TrainingOptions &get_options(void) const noexcept {
    return options_;
  }

private:
  TrainingOptions options_; /**< The training configurations */
};

/**
 * The ErisAggregatorBuilder class contains all the configuration parameters
 * for building an ErisAggregator.
 */
class ErisAggregatorBuilder final : public ErisServiceBuilder {
public:
  /**
   * It constructs a builder with the default parameters.
   */
  explicit ErisAggregatorBuilder(uint32_t fragment_id, size_t fragment_size);

  /**
   * Deletes an instance of an ErisAggregatorBuilder object.
   */
  virtual ~ErisAggregatorBuilder(void) = default;

  bool add_min_clients(uint32_t min_clients) noexcept;

  inline uint32_t get_min_client(void) const noexcept { return min_clients_; }
  inline uint32_t get_fragment_id(void) const noexcept { return fragment_id_; }
  inline size_t get_fragment_size(void) const noexcept {
    return fragment_size_;
  }

private:
  uint32_t fragment_id_;
  size_t fragment_size_;
  uint32_t min_clients_;
};
