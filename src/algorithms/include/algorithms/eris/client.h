#pragma once

#include "algorithms/eris/aggregator.grpc.pb.h"
#include "algorithms/eris/aggregator.h"
#include "algorithms/eris/builder.h"
#include "algorithms/eris/coordinator.h"
#include "algorithms/eris/coordinator.pb.h"
#include "algorithms/eris/split.h"
#include "erisfl/client.h"
#include <atomic>
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

  virtual ~ErisClient(void);
  void start(void) override;

  bool set_coordinator_rpc(const std::string &address);
  bool set_coordinator_subscription(const std::string &address);
  bool set_aggregator_config(const std::string &address, uint16_t submit_port,
                             uint16_t publish_port);

protected:
  bool join(void);

  std::mutex mu_;
  TrainingOptions options_;
  std::vector<void *> subscriptions_;
  std::vector<std::unique_ptr<eris::Aggregator::Stub>> submitters_;

  std::unique_ptr<ErisAggregator> aggregator_;
  std::unique_ptr<std::thread> aggregator_thread_;

private:
  bool register_aggregator(const FragmentInfo &aggregator) noexcept;
  void start_aggregator(uint32_t fragment_id) noexcept;
  void coordinator_subscribe(void) noexcept;

  std::string rpc_address_;
  std::string subscribe_address_;

  std::string aggr_address_;
  uint16_t aggr_rpc_port_;
  uint16_t aggr_publish_port_;

  void *zmq_ctx;
  void *coord_sub;

  std::unique_ptr<std::thread> coord_updater_;

  RandomSplit splitter;
  std::atomic_bool listening_;
};
