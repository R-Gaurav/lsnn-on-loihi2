#include "ldn_enc_to_spk_on_lmt_hard_coded_sig.h"
#include "predefs_CLdnEncToSpkOnLmtHCSigModel.h"

// Define the global LDN state vector with maximum ORDER = 128.
int32_t g_ldn_state[128] = {0};
int32_t g_volt[256] = {0}; // Set twice the ORDER number of spiking neurons.

int spk_guard(runState *rs) {
  return 1; // keep running the `run_spk()` every time-step.
}

// Hard-codede input signal.
int32_t input[140] = {
	-461, -11580, -15458, -17817, -17924, -14234,  -8935,  -7448,
        -5122,  -1956,  -1490,  -2015,  -1728,  -1266,  -2031,  -1401,
        -1455,  -1507,  -1296,  -1689,  -1932,  -1694,  -1493,  -1840,
        -1931,  -1740,  -1894,  -2263,  -1947,  -2843,  -2875,  -2432,
        -2706,  -2924,  -3153,  -2754,  -2677,  -2619,  -2291,  -2423,
        -2020,  -1897,  -1236,   -953,   -512,   -631,   -100,   -269,
          143,    254,    292,    508,    422,    923,    527,   1239,
         1054,    804,    735,   1002,   1398,   1344,   1663,   1829,
         1737,   1972,   1957,   2556,   2353,   2449,   2313,   2490,
         2542,   2688,   2805,   2844,   2726,   2358,   2614,   2519,
         2331,   1919,   1814,   1918,   1771,   1671,   1715,   1485,
         1683,   1932,   1524,   1384,    907,   1122,   1223,   1080,
         1403,   1718,   2403,   3525,   4806,   5154,   5873,   6965,
         8188,   8705,   8165,   7914,   7362,   6235,   5125,   4091,
         1981,     95,   -798,   -905,   -998,  -1043,  -1192,  -1051,
         -933,  -1321,  -1185,  -1303,  -1490,  -1612,  -1091,  -1052,
        -1182,   -665,    657,   3245,   3824,   3264,   2370,   1056,
          934,    506,   3790,    791
};

// Following function is called every time-step, which implies
void run_spk(runState *rs) {
  uint32_t spike_data[spk_out.size];

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
    Bpu[i] = Bp[i] * input[(rs->time_step-1)%140];
  }

  // Compute Apx[t] + Bpu[t].
  for (uint32_t i=0; i< *ORDER; i++) {
    int32_t state = Apx[i] + Bpu[i];
    g_ldn_state[i] = (
	(state > 0) ? (1 + (state-1)/ *scale_factor) : (state/ *scale_factor));
  }

  // Rate Encode the current time-step's ldn_state to the spikes.
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
