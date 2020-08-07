__kernel void sum(__global float4* a_g, __global const float4* b_g, __global float4* res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
