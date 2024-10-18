#include "ldn_enc_to_spk_on_lmt.h"
#include "predefs_CLdnEncToSpkOnLmtModel.h"

// Define the global LDN state vector with maximum ORDER = 128.
int32_t g_ldn_state[128] = {0};
int32_t g_volt[256] = {0}; // Set twice the ORDER number of spiking neurons.

int spk_guard(runState *rs) {
  return 1; // keep running the `run_spk()` every time-step.
}

int post_guard(runState *rs) {
  if (rs->time_step % ps_ts[0] == 1) // Presentation time of one sample is over.
    return 1;

  return 0;
}

void zero_out_global_arrays() {
  for(uint32_t i=0; i< ORDER[0]; i++)
    g_ldn_state[i] = 0;

  for(uint32_t i=0; i< 2*ORDER[0]; i++)
    g_volt[i] = 0;
}

void run_post_mgmt(runState *rs) {
  zero_out_global_arrays();
}

// Following function is called every time-step.
void run_spk(runState *rs) {

  if (rs->time_step % ps_ts[0] == 1) // Presentation time of one sample is over.
    zero_out_global_arrays();

  int32_t input[sig_inp.size];
  uint32_t spike_data[spk_out.size];

  recv_vec_dense(rs, &sig_inp, input); // Get u[t].

  // Note that the matrices Ap, Bp, encoders E, and ORDER are already defined
  // in Process. The variable ORDER can be accessed as *ORDER or ORDER[0].

  // Compute Ap * x[t] and Bp * u[t].
  int32_t Apx[*ORDER], Bpu[*ORDER];
  for(uint32_t i=0; i< *ORDER; i++) {
    int32_t sum = 0;
    for(uint32_t j=0; j< *ORDER; j++) {
      sum += (Ap[i][j] * g_ldn_state[j]);
    }
    Apx[i] = sum;
    Bpu[i] = Bp[i][0] * input[0];
  }

  // Compute Apx[t] + Bpu[t].
  for (uint32_t i=0; i< *ORDER; i++) {
    int32_t state = Apx[i] + Bpu[i];
    g_ldn_state[i] = (
	    //Ceil Integer Division
	    (state > 0) ? (1 + (state-1)/ *scale_factor) : (state/ *scale_factor)
	  );
  }

  // Rate Encode the current time-step's ldn_state to the spikes. Note that the
  // g_gain, g_bias, and g_v_thr are already defined in the the Process.
  for(uint32_t i=0; i< 2* *ORDER; i++) {
    int32_t J = *g_gain * E[i] * g_ldn_state[i/2] + *g_bias; // g_bias = 0 here.
    g_volt[i] = g_volt[i] + J; // Update the global voltage state of IF neurons.
    if(g_volt[i] > *g_v_thr) {
      spike_data[i] = 1;
      g_volt[i] = 0;
    }
    else {
      spike_data[i] = 0;
      if(g_volt[i] < 0)
	      g_volt[i] = 0; // Rectifiy the voltage.
    }
  }

  send_vec_dense(rs, &spk_out, spike_data);
}
