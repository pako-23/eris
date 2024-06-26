#pragma once

#include "algorithms/eris/builder.h"
#include "algorithms/eris/common.pb.h"
#include "spdlog/spdlog.h"
#include "zmq.h"
#include <grpcpp/impl/codegen/config_protobuf.h>
#include <grpcpp/impl/service_type.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <string>
#include <type_traits>

using grpc::Server;
using grpc::protobuf::Message;

/**
 * The ErisService class implements the communication services needed by the
 * eris federated training algorithm. In particular, it starts a gRPC server to
 * handle incoming requests, and a ZeroMQ socket to publish events.
 */
template <class T> class ErisService {
  static_assert(std::is_base_of<grpc::Service, T>::value,
                "The type should ineherit from grpc::Service");

public:
  /**
   * It constructs an ErisService object with the provided configurations, and
   * gRPC service arguments. Upon construction, the process will start listening
   * on the provided publish address and gRPC address.
   *
   * @param builder The builder class carrying all the configurations to build
   * an ErisService.
   * @param args The arguments that should be used to build gRPC service that
   * will handle incoming gRPC requests.
   */
  template <class... Args>
  explicit ErisService(const ErisServiceBuilder &builder, Args &&...args)
      : server_{nullptr}, service_{args...}, started_{false},
        listening_address_{builder.get_rpc_listen_address()} {
    grpc::ServerBuilder srv_builder;
    int port;
    char endpoint[255];
    size_t endpointlen = sizeof(endpoint);

    zmq_ctx = zmq_ctx_new();
    if (!zmq_ctx)
      throw std::bad_alloc();

    publisher = zmq_socket(zmq_ctx, ZMQ_PUB);
    if (!publisher)
      throw std::bad_alloc();

    zmq_bind(publisher, builder.get_pubsub_listen_address().c_str());
    zmq_getsockopt(publisher, ZMQ_LAST_ENDPOINT, &endpoint, &endpointlen);
    publish_port_ = atoi(strchr(strchr(endpoint, ':') + 1, ':') + 1);

    srv_builder.AddListeningPort(builder.get_rpc_listen_address(),
                                 grpc::InsecureServerCredentials(), &port);
    srv_builder.RegisterService(&service_);

    server_ = srv_builder.BuildAndStart();
    rpc_port_ = port;

    // Sleep for 200 milliseconds to prevent the slow joiner syndrome
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  /**
   * Deletes an instance of an ErisService object.
   */
  ~ErisService(void) noexcept {
    if (started_)
      stop();
    zmq_close(publisher);
    zmq_ctx_destroy(zmq_ctx);
  }

  /**
   * Stars the ErisService. In practice, it will start serving RPC
   * requests.
   */
  void start(void) noexcept {
    started_ = true;
    char endpoint[255];
    size_t endpointlen = sizeof(endpoint);

    zmq_getsockopt(publisher, ZMQ_LAST_ENDPOINT, &endpoint, &endpointlen);

    spdlog::info("listening RPC requests on {0}:{1} and publishing on {2}",
                 listening_address_.substr(0, listening_address_.find(':')),
                 rpc_port_, endpoint);
    server_->Wait();
  }

  /**
   * Stops the ErisService. In practice, it will stop serving RPC
   * requests.
   */
  void stop(void) noexcept {
    server_->Shutdown();
    started_ = false;
  }

  /**
   * It returns the port on which the ErisService publishes events.
   *
   * @return The port on which the service publishes events.
   */
  inline uint16_t get_publish_port(void) const noexcept {
    return publish_port_;
  }

  /**
   * It returns the port on which the ErisService serves gRPC requests.
   *
   * @return The port on which the service serves gRPC requests.
   */
  inline uint16_t get_rpc_port(void) const noexcept { return rpc_port_; }

  /**
   * It publishes a message over the ZeroMQ socket.
   *
   * @param message The message that should be published over the ZeroMQ socket.
   * It serializes the message with Protobuf.
   * @return It returns true if the message was correctly published; otherwise,
   * it returns false.
   */
  bool publish(const Message &message) noexcept {
    zmq_msg_t msg;

    zmq_msg_init_size(&msg, message.ByteSizeLong());
    message.SerializeToArray(zmq_msg_data(&msg), message.ByteSizeLong());
    bool ret = zmq_msg_send(&msg, publisher, 0) > 0;
    zmq_msg_close(&msg);

    return ret;
  }

private:
  std::unique_ptr<Server> server_;      /**< The gRPC server */
  void *zmq_ctx;                        /**< The ZeroMQ socket context */
  void *publisher;                      /**< The ZeroMQ publisher socket */
  uint16_t rpc_port_;                   /**< The gRPC listening port */
  uint16_t publish_port_;               /**< The ZeroMQ listening port */
  T service_;                           /**< The gRPC registered service */
  bool started_;                        /**< If the service is started or no */
  const std::string listening_address_; /**< The gRPC listening address */
};
