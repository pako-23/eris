#pragma once

#include <string>

/**
 * Checks if address contains a valid IPv4 address.
 *
 * @param address A string containing the IPv4 address.
 * @return true if the address contains a valid IPv4 address; otherwise false.
 */
bool valid_ipv4(const std::string &address);

/**
 * Checks if address contains a valid aggregator submission address.
 * An aggregator submission address is valid if it has the form
 * <address>:<port>, where address must be a valid IPv4 address and port must be
 * a valid port number. Also, address cannot be 0.0.0.0.
 *
 * @param address A string containing the submission address of an aggregator.
 * @return true If the address contains a valid aggregator submission address;
 * otherwise false.
 */
bool valid_aggregator_submit(const std::string &address);

/**
 * Checks if address contains a valid aggregator publishing address.
 * An aggregator publishing address is valid if it has the form
 * tcp://<address>:<port>, where address must be a valid IPv4 address and port
 * must be a valid port number. Also, address cannot be 0.0.0.0, not *.
 *
 * @param address A string containing the publishing address of an aggregator.
 * @return true If the address contains a valid aggregator submission address;
 * otherwise false.
 */
bool valid_aggregator_publish(const std::string &address);
