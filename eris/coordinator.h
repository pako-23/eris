#pragma once

class Coordinator {
public:
  virtual ~Coordinator() {}
  virtual void run(void) = 0;
};


class ErisCoordinator : public Coordinator {
  void run(void) override;
};
