# coral-dev-opencl
My experiments on using OpenCL on the Google Coral Deb board

### Prerequisites

### Coral Setup


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
          #network 10.0.0.1
          #broadcast 10.0.0.255
          #gateway 10.0.0.1
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
