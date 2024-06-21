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
 * Checks if aggregator contains a valid aggregator address.
 * An aggregator address is valid if it has the form <address>:<port>, where
 * address must be a valid IPv4 address and port must be a valid port number.
 *
 * @param aggregator A string containing the address of an aggregator.
 * @return true if the address contains a valid aggregator address; otherwise
 * false.
 */
bool valid_aggregator(const std::string &aggregator);
