# coral-dev-opencl

My experiments on using OpenCL on the Google Coral Dev board GPU associated to ML models on the TPU

### Prerequisites

 - Linux computer (referred to below as "host") with python3 and pip3
 - USB-A to USB-micro-B cable (to connect your PC to the board's serial port)
 - USB-A to USB-C cable (to connect your PC to the board's data port)
 - 2-3A (5V) USB Type-C power supply
 - Ethernet cable or Wi-Fi connection

### Coral Setup

#### Prepare the host:

 - Install a serial terminal
   ````
   sudo apt-get install screen
   ````
 
 - Get `fastboot` from the Android tools from `https://developer.android.com/studio/releases/platform-tools#downloads
   ````
   mkdir -p ~/.local/bin
   sudo mv ~/Downloads/platform-tools/fastboot ~/.local/bin/
   ````
 
 - Get MDT (Mendel Dev Tools)
   ````
   pip3 install --user mendel-development-tool
   ````
 
 - Sometimes needed for the serial console (not on my rpi)
   ````
   sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0525\", MODE=\"0664\", \
   GROUP=\"plugdev\", TAG+=\"uaccess\"' >> /etc/udev/rules.d/65-edgetpu-board.rules"

   sudo udevadm control --reload-rules && sudo udevadm trigger
   ````

#### Connect thru USB OTG

  ````
  mdt devices
  mdt shell
  ````

#### Download and install Mendel distribution

  ````
  cd ~/Downloads

  curl -O https://mendel-linux.org/images/enterprise/eagle/enterprise-eagle-20200724205123.zip

  unzip enterprise-eagle-20200724205123.zip \
  && cd enterprise-eagle-20200724205123

  bash flash.sh
  ````

#### Setup Wifi connection

````
nmtui
nmcli connection show
````

#### Set SD Card as a drive

 - To create a file swap REQUIRED to compile pyopencl
   ````
   sudo umount /dev/mmcblk1
   sudo mkdir /mnt/SD
   sudo mkfs -t ext4 /dev/mmcblk1
   sudo mount -t ext4 /dev/mmcblk1 /mnt/SD 

   sudo apt-get install exuberant-ctags
   sudo apt-get install dphys-swapfile

   sudo dphys-swapfile setup
   sudo chmod 0600 /mnt/SD/swap
   sudo dphys-swapfile swapon
   watch -n1 free
   ````

- To create a storage drive (non-journalized)
  ````
  sudo umount /dev/mmcblk1
  sudo mkdir /mnt/SD
  sudo fdisk -l
  sudo mkfs -t ext2 /dev/mmcblk1
  sudo mount -t ext2 /dev/mmcblk1 /mnt/SD
  sudo nano /etc/fstab
  ````

 - Add: `/dev/mmcblk1 /mnt/SD ext2 defaults 0 3`

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

  dmesg | grep ttyUSB
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
  ````

##### Demo Streaming

`edgetpu_demo --stream`

##### Demo classification

 - Try tflite demo:
    ````
    sudo apt-get install git
    mkdir coral && cd coral
    git clone https://github.com/google-coral/tflite.git
    cd tflite/python/examples/classification
    bash install_requirements.sh
    python3 classify_image.py --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --labels models/inat_bird_labels.txt --input images/parrot.jpg
    ````

 - Try EdgeTPU Python API:

   ````
   sudo apt-get install edgetpu-examples
   cd /usr/share/edgetpu/examples/

   python3 classify_image.py --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --label models/inat_bird_labels.txt --image images/parrot.jpg
   python3 object_detection.py --model models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite --label models/coco_labels.txt --input images/grace_hopper.bmp --output ${HOME}/object_detection_results.jpg
   python3 object_detection.py --model models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite --input images/grace_hopper.bmp --output ${HOME}/face_detection_results.jpg
   python3 examples/semantic_segmentation.py --model models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite --input models/bird.bmp --keep_aspect_ratio --output ${HOME}/segmentation_result.jpg
   ````

### Setup OpenCL

  ````
  sudo apt-get install clinfo opencl-c-headers opencl-clhpp-headers
  sudo apt-get install ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev
  clinfo
```` 

- Patch the opencl headers (not the best way but I did not find another) by adding at the beginning:
  ````
  sudo nano /usr/include/CL/cl_version.h
  ````

  ````
  /* patch! */
  #define CL_TARGET_OPENCL_VERSION 120
  ````

- Set the SD card swap! 
  ````
  sudo dphys-swapfile swapon
  ````

- Install the libraries
  ````
  sudo apt-get install python3-dev libatlas-base-dev
  pip3 install rpyc numpy==1.19.0 mako pybind11
  pip3 install pyopencl
  sudo dphys-swapfile swapoff
  ````

- If doesnt work:
  ````
  git clone https://github.com/inducer/pyopencl.git
  cd pyopencl
  python configure.py --cl-pretend-version=1.2
  make
  #sudo make install
  rm -Rf build
  pip3 install .
  ````

### Cluster performance

#### Network

Check the bandwidth between each node:

 - Create a iperf server on one node, `opencl1` for example
  ```
  sudo apt-get install iperf
  iperf -s
  ```

 - Create an iperf client on each other node:
  ```
  sudo apt-get install iperf
  iperf -c opencl1
  ```

On my cluster, as expected I got arounf 94.4Mb/s to a raspberry pi 3B and 932Mb/s to the Coral Dev board (from a laptop of course)

#### OpenCL

Reminder, the openCL Coral implementation is for the Vivante GC7000Lite GPU, **NOT FOR THE TPU**.
Notes:
- GC7000Lite local memory is only 32KB, so a scratchpad

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
