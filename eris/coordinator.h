#pragma once

class Coordinator {
 public:
  virtual ~Coordinator() {}
  virtual void run(void) = 0;
};
