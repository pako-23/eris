#pragma once

class Client {
public:
  virtual ~Client() {};
  
  virtual void run(void);
  virtual void get_parameters(void) =0;
  virtual void fit(void) =0;
  virtual void evaluate(void) =0;
};
