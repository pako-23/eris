#pragma once

class Coordinator {
public:
  virtual ~Coordinator(void) = default;
  virtual void start(void) = 0;
  virtual void stop(void) = 0;
};
