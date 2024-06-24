#pragma once

#include "algorithms/eris/coordinator.pb.h"
#include <cstdint>
#include <string>

#define DEFAULT_ERIS_RPC_PORT 50051
#define DEFAULT_ERIS_PUBSUB_PORT 5555

/**
 * The ErisServiceBuilder is a base class for building an eris service builder.
 * A builder will carry configuration parameters for building an eris service.
 * The ErisCoordinator is an example fo an eris service.
 */
class ErisServiceBuilder {
protected:
  /**
   * It constructs a builder with the default parameters.
   */
  explicit ErisServiceBuilder(void);

public:
  /**
   * Deletes an instance of an ErisServiceBuilder object.
   */
  virtual ~ErisServiceBuilder(void) = default;

  /**
   * It sets the port on which the service should listen for incoming RPC
   * requests.
   *
   * @param port The RPC listening port.
   * @return It returns true if it manages to successfully set the RPC port;
   * otherwise it returns false. In practice, it always returns true.
   */
  bool add_rpc_port(uint16_t port);
  bool add_rpc_listen_address(const std::string &address);

  /**
   * It sets the port on which the service should publish events for new
   * aggregators joining to the clients.
   *
   * @param port The Pub-Sub publishing port.
   * @return It returns true if it manages to successfully set the Pub-Sub port;
   * otherwise it returns false. In practice, it always returns true.
   */
  bool add_pubsub_port(uint16_t port);
  bool add_pubsub_listen_address(const std::string &address);

  inline const std::string get_rpc_listen_address(void) const {
    return rpc_listen_address_ + ":" + std::to_string(rpc_port_);
  }

  inline const std::string get_pubsub_listen_address(void) const {
    std::string address =
        pubsub_listen_address_ == "0.0.0.0" ? "*" : pubsub_listen_address_;
    return "tcp://" + address + ":" + std::to_string(pubsub_port_);
  }

protected:
  std::string rpc_listen_address_;
  uint16_t rpc_port_;

  std::string pubsub_listen_address_;
  uint16_t pubsub_port_;
};

class ErisCoordinatorBuilder final : public ErisServiceBuilder {
public:
  explicit ErisCoordinatorBuilder(void);

  /**
   * Deletes an instance of an ErisCoordinatorBuilder object.
   */
  virtual ~ErisCoordinatorBuilder(void) = default;

  bool add_rounds(uint32_t rounds);
  bool add_splits(uint32_t splits);
  bool add_min_clients(uint32_t min_clients);
  bool add_split_seed(uint32_t split_seed);

  inline const eris::TrainingOptions &get_options(void) const {
    return options_;
  }

private:
  eris::TrainingOptions options_;
};
