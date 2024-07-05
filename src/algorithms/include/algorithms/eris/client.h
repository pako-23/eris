#pragma once

#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/split.h"
#include "erisfl/client.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using eris::TrainingOptions;

class ErisClient : public Client {
public:
  explicit ErisClient(void);

  virtual ~ErisClient(void) = default;
  bool start(void) override;

  bool set_coordinator_rpc(const std::string &address);
  bool set_coordinator_subscription(const std::string &address);
  bool set_aggregator_config(const std::string &address, uint16_t submit_port,
                             uint16_t publish_port);

protected:
  class ClientState {
  public:
    explicit ClientState(void);
    ~ClientState(void);

    bool join(const ErisClient *client, const std::string &rpc_address,
              const std::string &subscribe_address,
              const std::string *listen_address = nullptr,
              const uint16_t *rpc_port = nullptr,
              const uint16_t *publish_port = nullptr);

    bool configure(const ErisClient *client, const InitialState &state);

    bool submit_weights(const std::vector<double> &parameters, uint32_t round);

    inline void unlock(void) { mu_.unlock(); }
    inline void lock(void) { mu_.lock(); }

    inline const TrainingOptions &get_options(void) const noexcept {
      return options_;
    }

    inline const std::vector<void *> &get_subscriptions(void) const noexcept {
      return subscriptions_;
    }

    inline const std::vector<std::unique_ptr<eris::Aggregator::Stub>> &
    get_submitters(void) const noexcept {
      return submitters_;
    }

    inline bool is_aggregator(void) const noexcept {
      return aggregator_ != nullptr && aggregator_thread_ != nullptr;
    }

    bool register_aggregator(const FragmentInfo &aggregator) noexcept;

  private:
    void start_aggregator(uint32_t fragment_id,
                          const std::string *listen_address,
                          const uint16_t *rpc_port,
                          const uint16_t *publish_port) noexcept;
    void coordinator_subscribe(const std::string &subscribe_address) noexcept;

    void *zmq_ctx;
    void *coord_sub;
    std::atomic_bool coord_subscribed_;

    std::mutex mu_;
    std::condition_variable aggregator_joined_;
    TrainingOptions options_;
    RandomSplit splitter;

    std::vector<void *> subscriptions_;
    std::vector<std::unique_ptr<eris::Aggregator::Stub>> submitters_;

    std::unique_ptr<std::thread> coord_updater_;

    std::unique_ptr<ErisAggregator> aggregator_;
    std::unique_ptr<std::thread> aggregator_thread_;
  };

private:
  std::string rpc_address_;
  std::string subscribe_address_;

  std::string aggr_address_;
  uint16_t aggr_rpc_port_;
  uint16_t aggr_publish_port_;

  ClientState state_;
};
