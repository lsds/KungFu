#include <kungfu.h>
#include <libkungfu-comm.h>

void kungfu_run_main() { GoKungfuRunMain(); }
void kungfu_run_send_begin() { GoKungfuRunSendBegin(); }
void kungfu_run_send_end() { GoKungfuRunSendEnd(); }
void kungfu_run_send_trainend() { GoKungfuRunSendTrainend(); }
void kungfu_run_send_epoch() { GoKungfuRunSendEpoch(); }
