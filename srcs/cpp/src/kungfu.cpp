#include <kungfu.h>
#include <libkungfu-comm.h>

void kungfu_run_main() { GoKungfuRunMain(); }
void kungfu_run_send_signal(int sig) { GoKungfuSignalSend(GoInt(sig)); }
