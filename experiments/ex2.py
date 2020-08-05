import numpy as np
import pyopencl as cl
import time

vector_size = 10000000
vector_type = np.float32

print("Generate {} random numbers of type {}".format(vector_size, str(vector_type)))
a_np = np.random.rand(vector_size).astype(vector_type)
b_np = np.random.rand(vector_size).astype(vector_type)
res_np = np.empty_like(a_np).astype(vector_type)
print("Buffer types: {}, {}, {} of size: {} bytes".format(a_np.dtype, b_np.dtype, res_np.dtype, a_np.nbytes))

print("---------------------------------------------------------------------------")

# get platforms for each cluster
print("Available platforms on the Clusters:")

platforms = cl.get_platforms()

for index, platform in enumerate(platforms):
    print("{}\tPlatform: {}".format(index, platform))
    print("\tName: {}".format(platform.name))
    print("\tProfile: {}".format(platform.profile))
    print("\tVendor: {}".format(platform.vendor))
    print("\tVersion: {}".format(platform.version))

print("---------------------------------------------------------------------------")

# create openCL context on platform rpi1, first device
print("Getting node for platform")

device_nb = 0
print("Create OpenCL context on device {}".format(device_nb))
ctx = cl.Context(dev_type=cl.device_type.ALL,  properties=[(cl.context_properties.PLATFORM, platforms[device_nb])])
device = ctx.devices[0]
float_vector_size = device.preferred_vector_width_float

print("Device {} properties:".format(device))
print("\tPrefered float vector size: {}".format(float_vector_size))
print("\tVersion: {}".format(device.version))
print("\tVendor: {}".format(device.vendor_id))
print("\tProfile: {}".format(device.profile))
print("\topencl_c_version: {}".format(device.opencl_c_version))
print("\tmax_compute_units: {}".format(device.max_compute_units))
print("\tmax_clock_frequency: {}".format(device.max_clock_frequency))
print("\tlocal_mem_size: {}".format(device.local_mem_size))
print("\tglobal_mem_size: {}".format(device.global_mem_size))
print("\textensions: {}".format(device.extensions))

print("Create OpenCL queue")
queue = cl.CommandQueue(ctx)

print("Copy data to device buffers")
mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

print("Reading kernel file: demo_float.cl")
with open("demo_float.cl", "r") as f_kernel:
    kernel = f_kernel.read()

print("Compiling kernel")
prg = cl.Program(ctx, kernel).build()

print("Executing computation")
t0 = time.perf_counter_ns()
#prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
knl = prg.sum
knl.set_args(a_g, b_g, res_g)
local_work_size = None
#local_work_size = (10,)

t1 = time.perf_counter_ns()
ev = cl.enqueue_nd_range_kernel(queue=queue, kernel=knl, global_work_size=(vector_size,), local_work_size=local_work_size)
t0_enqueue = time.perf_counter_ns()
ev.wait()
t2 = time.perf_counter_ns()

t3 = time.perf_counter_ns()
cl.enqueue_copy(queue, res_np, res_g)
t4 = time.perf_counter_ns()

# Check on CPU with Numpy:
print("Computing on the host using numpy")
t5 = time.perf_counter_ns()
res_local = a_np + b_np
t6 = time.perf_counter_ns()
print("Local type:", res_local.dtype)

print("---------------------------------------------------------------------------")
print("Comparing results")
print("Difference   : {}".format(res_np - res_local))
print("A            : {}".format(a_np))
print("B            : {}".format(b_np))
print("Result OpenCL: {}".format(res_np))
print("Result Numpy : {}".format(res_local))

print("Checking the norm between both: {}".format(np.linalg.norm(res_np - res_local)))
print("Checking results are mostly the same: ", np.allclose(res_np, res_local))

print("---------------------------------------------------------------------------")
print("Time to compute using opencl: {} ms".format((t4-t0)/1000000))
print("Time to compute using numpy: {} ms".format((t6-t5)/1000000))

