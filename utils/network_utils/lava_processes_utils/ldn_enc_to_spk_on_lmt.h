#include "nxsdk.h"

int spk_guard(runState *rs);
int post_guard(runState *rs);
void run_post_mgmt(runState *rs);
void run_spk(runState *rs);

// Don't need the following, but still!
void zero_out_global_arrays();
