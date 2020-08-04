# coral-dev-opencl
My experiments on using OpenCL on the Google Coral Deb board

### Prerequisites

### Coral Setup

#### Connect thru USB OTG

````
mdt devices
mdt shell
````

#### Setup Wifi connection

````
nmtui
nmcli connection show
````

#### Set SD Card as a drive

````
sudo mkdir /mnt/SD
sudo fdisk -l
sudo mkfs -t ext2 /dev/mmcblk1
sudo mount -t ext2 /dev/mmcblk1 /mnt/SD
sudo nano /etc/fstab
````

Add: `/dev/mmcblk1 /mnt/SD ext2 defaults 0 3`

#### Set SSH keys

I did not find a way to revert Mendel sshd configuration to accept user/passwords and not only public keys certificates.

So, when `mdt shell` worked and you still connected using the USB Data Port, from your **host**, type

 ````
 ssh-keygen
 mdt pushkey ~/.ssh/id_rsa.pub
 ssh mendel@192.168.100.2
 ````

#### Connect thru Serial Shell if needed

From the host
````
sudo apt-get install screen
pip3 install --user mendel-development-tool

dmesg
screen /dev/ttyUSB0 115200
````

Default login/password: mendel

#### Static IP on the OpenCL private network

 - Edit as sudoer `/etc/network/interfaces` and modify:
  ````
  # interfaces(5) file used by ifup(8) and ifdown(8)
  # Include files from /etc/network/interfaces.d:
  source-directory /etc/network/interfaces.d

  allow-hotplug eth0
  iface eth0 inet static
          address 10.0.0.4
          netmask 255.255.255.0
  ````

 - Restart the network service: `sudo systemctl restart networking`
 - Check the routes with `ip addr show eth0` 
 
 ````
   2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
      link/ether 7c:d9:5c:b2:75:f8 brd ff:ff:ff:ff:ff:ff
      inet 10.0.0.4/24 brd 10.0.0.255 scope global eth0
         valid_lft forever preferred_lft forever
      inet6 fe80::7ed9:5cff:feb2:75f8/64 scope link
         valid_lft forever preferred_lft forever
 ````
 
  - then  `ip route show`
  
  ````
  default via 192.168.1.1 dev wlan0 proto dhcp metric 600
  10.0.0.0/24 dev eth0 proto kernel scope link src 10.0.0.4
  192.168.1.0/24 dev wlan0 proto kernel scope link src 192.168.1.43 metric 600
  192.168.100.0/24 dev usb0 proto kernel scope link src 192.168.100.2 metric 100
  192.168.101.0/24 dev usb1 proto kernel scope link src 192.168.101.2 metric 101 linkdown
  ````

#### Install TPU libraries

````
sudo apt-get update
sudo apt-get dist-upgrade
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl
edgetpu_demo --stream
sudo apt-get install git
mkdir coral && cd coral
git clone https://github.com/google-coral/tflite.git
cd tflite/python/examples/classification
bash install_requirements.sh
python3 classify_image.py --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --labels models/inat_bird_labels.txt --input images/parrot.jpg
````

### Setup OpenCL

````
sudo apt-get install clinfo opencl-c-headers opencl-clhpp-headers opencl-headers opencl-headers ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev
clinfo
```` 

````
sudo apt-get install python3-dev
pip3 install rpyc numpy
pip3 install pyopencl
````


### OpenCL performance

Reminder, the openCL Coral implementatinon is for the Vivandi G7000Lite GPU, NOT FOR THE TPU.

clpeak:
````
Platform: Vivante OpenCL Platform
  Device: Vivante OpenCL Device GC7000L.6214.0000
    Driver version  : OpenCL 1.2 V6.4.2.256507 (Linux ARM64)
    Compute units   : 1
    Clock frequency : 800 MHz

    Global memory bandwidth (GBPS)
      float   : 3.38
      float2  : 4.88
      float4  : 4.98
      float8  : 4.71
      float16 : 3.57

    Single-precision compute (GFLOPS)
      float   : 4.65
      float2  : 10.18
      float4  : 21.27
      float8  : 22.21
      float16 : 24.86

    Half-precision compute (GFLOPS)
      half   : 4.65
      half2  : 10.18
      half4  : 21.27
      half8  : 22.20
      half16 : 24.85

    No double precision support! Skipped

    Integer compute (GIOPS)
      int   : 5.67
      int2  : 5.99
      int4  : 6.37
      int8  : 6.34
      int16 : 6.32

    Integer compute Fast 24bit (GIOPS)
      int   : 5.67
      int2  : 5.99
      int4  : 6.37
      int8  : 6.34
      int16 : 6.32

    Transfer bandwidth (GBPS)
      enqueueWriteBuffer              : 1.97
      enqueueReadBuffer               : 0.11
      enqueueWriteBuffer non-blocking : 2.05
      enqueueReadBuffer non-blocking  : 0.12
      enqueueMapBuffer(for read)      : 99.55
        memcpy from mapped ptr        : 0.12
      enqueueUnmap(after write)       : 103.91
        memcpy to mapped ptr          : 2.03

    Kernel launch latency : 206.17 us
````

To compare with the Raspberry Pi 3 Videocore IV performance:

````
Platform: OpenCL for the Raspberry Pi VideoCore IV GPU
  Device: VideoCore IV GPU
    Driver version  : 0.4.9999 (Linux ARM)
    Compute units   : 1
    Clock frequency : 300 MHz

    Global memory bandwidth (GBPS)
clCreateBuffer (-5)
      Tests skipped

    Single-precision compute (GFLOPS)
      float   : 0.60
      float2  : 1.13
      float4  : 2.00
      float8  : 3.31
      float16 : 4.60

    No half precision support! Skipped

    No double precision support! Skipped

    Integer compute (GIOPS)
      int   : 0.16
      int2  : 0.30
      int4  : 0.60
      int8  : 0.77
      int16 : 1.25

    Integer compute Fast 24bit (GIOPS)
      int   : 0.57
      int2  : 1.02
      int4  : 1.73
      int8  : 2.51
      int16 : 3.27

    Transfer bandwidth (GBPS)
      enqueueWriteBuffer              : 1.22
      enqueueReadBuffer               : 0.25
      enqueueWriteBuffer non-blocking : 1.22
      enqueueReadBuffer non-blocking  : 0.25
      enqueueMapBuffer(for read)      : 1838.60
        memcpy from mapped ptr        : 0.24
      enqueueUnmap(after write)       : 2191.31
        memcpy to mapped ptr          : 1.22

    Kernel launch latency : 30.27 us
````
