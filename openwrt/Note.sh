#21年10月底，用了一个周末把姚俊峰的老笔记本改造成了openwrt软路由。
#之前用的是手机小米5手机热点+TP link的无线路由覆盖。网速比较慢，接入联接速度后比较卡顿。
#改造后，网速提升很快，接入多个设备也不卡。笔记本功耗很低，基本听不到风扇转动。

#网路的结构是：

#小米手机 ====》笔记本（软路由，DHCP，NAT网络转发，AdGuardHome) ====>内网用户
#                |（lan）
#                |
#                |
#                |（lan）
#               TP-Link(无线AP) ====》内网用户

#TP-Link 设置， 静态地址 192.168.1.1，关闭DHCP， 和笔记本LAN - LAN， 无线接入打开。设置TP—link 可以从无线端接入。

#软路由设置： BR——LAN ： 静态地址， IP 192.168.1.2， DHCP 打开。
            #USB——Tethering： DHCP Client，UBS口联接手机。（手机关闭蓝牙，关闭锁屏，打开USB调试模式）

#刷系统
    #镜像源
        #Openwrt官方网站 
        #Koolshare 已经停止更新
        #恩山 Lean （功能比较强大）
        #youtube esir 

        #squash / ext4 （没有恢复出厂功能）

    #刷机工具
        #试图用WinPE刷入，但是rufus只能刷u盘，Win32image 打不开。 最后拆硬盘，用台式机刷入。BalenaEtcher
        #tar包需要解开。不然报错。
        #如果能用image builder自己做一个带USB网络共享的包，后面可以省好多问题。
        #如果刷机前/时解决了空间问题。后面可以省很多时间


#USB热点分享
        #按照官方教程一步一步做，很简单。难点在于，没有USB之前，怎么上网。
            #用过两种办法，手机开热点，无线路由中继覆盖。然后无线路由Lan-Lan笔记本。笔记本中加interface Wan （eth0）
            #手机USB共享网络给台式机，台式机共享给网口，网口再连无线路由Wan口。无线路由Lan对笔记本Lan。 笔记本新建interface Wan （eth0）并暂停bg-Lan。上网安装完USB包后。重启。 Bg-lan就会恢复。（这样就可以重新联接到luci）


#扩容
    #分区 再硬盘上新建分区
    #挂载 基本上用了 esir 的教程，youtube 搜索 esir openwrt overlay。 挂载用命令行 mount /dev/sda3 /overlay， 并写入 vim /etc/rc.local ， 每次重启后都会执行。
    
        #官方的方法是扩大分区。我觉得风险比较大。失败了就变砖，还得重新刷。
        #用外挂分区的好处是，如果系统崩溃了，你的overlay不会丢。overlay里有你所有的更改，插件。重装时，只需要重新刷入标准包。（甚至还可以恢复出厂） 然后再重新挂overlay。这样设置，和软件包就都回来了。

#AdGuardHome

    #可以去广告，控制内网用户访问网页，强制安全搜索，过滤成人内容。
    #安装比较简单。

#笔记本安装ubuntu

#将来可以跑虚拟机和Docker，

#公司的小主机替下来可以装个NAS

#千兆路由器

#交换机

#常用命令

opkg update

opkg install xxx

opkg update && opkg install XXX


    uci show network
    uci set fstab.mount xxx = ""
    uci commit fstab
    uci -q delete fstab.overlay

    reboot


#-#-#-#-#-#-#-#-#-#-#-#-#-#-mount overlay #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
#硬盘增加分区，+ 挂载overlay流程

opkg update && opkg install cfdisk

cfdisk

#Free Space  / New / primary / write / yes

block info


# /dev/loop0: UUID="5de7e234-6913-43b6-8e3a-d26f587f76d5" LABEL="rootfs_data" VERSION="1.14" MOUNT="/overlay" TYPE="f2fs"
# /dev/sda1: UUID="84173db5-fa99-e35a-95c6-28613cc79ea9" LABEL="kernel" VERSION="1.0" MOUNT="/boot" TYPE="ext4"
# /dev/sda2: UUID="488a811c-6314f3f2-5e697022-95d0cdbb" VERSION="4.0" MOUNT="/rom" TYPE="squashfs"
# /dev/sda3: UUID="3cf63333-2e59-432b-8839-ac7cf5b825d1" VERSION="1.0" TYPE="ext4"


mkfs.ext4 /dev/sda3

mkdir /mnt/sda3

mount /dev/sda3 /mnt/sda3

cp -r /overlay/* /mnt/sda3

mount /dev/sda3 /overlay


vim /etc/rc.local （开机的自动批处理）

    i键，开始编辑
    Esc 退出编辑
    ：w保存
    ：q退出


    mount /dev/sda3 /overlay
    usbmuxd
    /opt/AdGuardHome/AdGuardHome #如果限制性这个，后面的语句就无法执行了

    exit 0


cat /etc/rc.local

reboot

df -h

# Filesystem                Size      Used Available Use% Mounted on
# /dev/root                 4.0M      4.0M         0 100% /rom
# tmpfs                     2.9G    232.0K      2.9G   0% /tmp
# /dev/loop0               98.1M     96.7M      1.3M  99% /overlay
# overlayfs:/overlay       98.1M     96.7M      1.3M  99% /
# /dev/sda1                15.7M      4.8M     10.7M  31% /boot
# /dev/sda1                15.7M      4.8M     10.7M  31% /boot
# tmpfs                   512.0K         0    512.0K   0% /dev

#-#-#-#-#-#-#-#-#-#-#-#-#-#- end of mount overlay #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-


#-#-#-#-#-#-#-#-#-AdGuardHome install#-#-#-#-#-#-#-#-
ssh root@192.168.1.1
opkg update && opkg install wget
mkdir /opt/ && cd /opt
#下载前检查版本
wget -c https://github.com/AdguardTeam/AdGuardHome/releases/download/v0.101.0/AdGuardHome_linux_armv5.tar.gz
tar xfvz AdGuardHome_linux_armv5.tar.gz
rm AdGuardHome_linux_armv5.tar.gz
#Either just run it:
/opt/AdGuardHome/AdGuardHome 

#or install it directly with:
/opt/AdGuardHome/AdGuardHome -s install

#DNS forwarding, after install the AdGuardHome , need to set up the DNS forwarding
# 在网络，==》 DHCP/DNS ==》基本设置 ==》 DNS转发 下输入 192.168.0.1#535

#在防火墙 自定义规则 加入 ，防止规则漏网
iptables -t nat -A PREROUTING -i br-lan -p udp --dport 53 -j DNAT --to 192.168.1.1:5353
iptables -t nat -A PREROUTING -i br-lan -p tcp --dport 53 -j DNAT --to 192.168.1.1:5353

#编辑脚本，让AdGuardHome 每次重启都自动启动
vim /etc/rc.local

#i for start writing
/opt/AdGuardHome/AdGuardHome
#Esc
#:w (save)
#:q (quit)

#-#-#-#-#-#-#-#-#-AdGuardHome install end #-#-#-#-#-#-#-#-

#USB tethering

opkg update
opkg install kmod-usb-net-rndis

opkg update
opkg install kmod-nls-base kmod-usb-core kmod-usb-net kmod-usb-net-cdc-ether kmod-usb2

opkg update
opkg install kmod-usb-net-ipheth usbmuxd libimobiledevice usbutils
 
# Call usbmuxd
usbmuxd -v
 
# Add usbmuxd to autostart
sed -i -e "\$i usbmuxd" /etc/rc.local

#安装好后，别忘了再路由器里增加interface




#装了eSir 的固件，然后自己配置了Adhomeguard，运行出现了问题。于是决定自己用rom builder 写一个镜像刷机。

#image builder 挺好用的，照着官方教程一步步做就行。


#-#-#-#-#-#-#-#-#-#- #manually add interface #-#-#-#-#-#-#-#-#-#-#-#-#-#-
#这个代码可以手动开启usb 上网的interface

uci set network.wan=interface
uci set network.wan.ifname='USB_Te'
uci set network.wan.proto='dhcp'
uci set network.wan.device='usb0'
uci commit network
/etc/init.d/network restart

uci show network

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-


#image builder code

# 官方教程挺好用，跟着照做就行
# 1. 在ubuntu装包

sudo apt-get update

sudo apt install build-essential libncurses5-dev libncursesw5-dev \
zlib1g-dev gawk git gettext libssl-dev xsltproc rsync wget unzip python

# 2.去页面下载包 X86 应该是这个包 https://downloads.openwrt.org/snapshots/targets/x86/64/openwrt-imagebuilder-x86-64.Linux-x86_64.tar.xz
# 这个包刷上的系统应该是最纯净的系统。啥也没有，纯净到连网页设置界面的LUCI都没有
# 所以要提前准备package 列表

# 先解包

tar -J -x -f openwrt-imagebuilder-*.tar.xz
cd openwrt-imagebuilder-*/

# 3.设置变量

make info #检查可用的profile

PROFILE="generic" #x86应该是这个

PACKAGES="kmod-usb-net-rndis kmod-nls-base kmod-usb-core kmod-usb-net kmod-usb-net-cdc-ether kmod-usb2 kmod-usb-net-ipheth usbmuxd libimobiledevice usbutils"
#以上是usb 共享上网，加luci 所需要的包，但是还是远远不够。

echo $(opkg list-installed | sed -e "s/\s.*$//") #应该提前在默认可以运行的固件里跑这个命令。获取包的列表。

# 3.5 copy existing config from current router

mkdir -p files/etc/config
scp root@192.168.0.1:/etc/config/network files/etc/config/
scp root@192.168.0.1:/etc/config/wireless files/etc/config/
scp root@192.168.0.1:/etc/config/firewall files/etc/config/

#romote access to router

ssh root@192.168.0.1

# 4.开搞
for X86 profile is not required.
#make image PROFILE="profile-name" PACKAGES="pkg1 pkg2 pkg3 -pkg4 -pkg5 -pkg6" FILES="files"

make image PACKAGES="kmod-usb-net-rndis kmod-nls-base kmod-usb-core kmod-usb-net kmod-usb-net-cdc-ether kmod-usb2 kmod-usb-net-ipheth usbmuxd libimobiledevice usbutils luci kmod-fs-ext4" FILES="files" CONFIG_TARGET_ROOTFS_PARTSIZE=1024 #looks like the 1024 is for 0.1M ???

# 5.清理
make clean

# 6.The built image will be found under the subdirectory ./bin/targets/<target>/generic or look inside ./build_dir/ for a files *-squashfs-sysupgrade.bin and *-squashfs-factory.bin (e.g. /build_dir/target-mips_24kc_musl/linux-ar71xx_tiny/tmp/openwrt-18.06.2-ar71xx-tiny-tl-wr740n-v6-squashfs-factory.bin)


# 7.copy the file to an linux system, and write it into vm.
#unzip the file 

gzip -d openwrt-21.02.1-x86-64-generic-ext4-combined.img.gz

#decompress the file .gz to .img

# 8.write it into a prepared hard disk file

lsblk #check the hard disk address

'''
NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
loop0    7:0    0     4K  1 loop /snap/bare/5
loop1    7:1    0 670.8M  1 loop /snap/pycharm-professional/269
loop2    7:2    0 144.6M  1 loop /snap/chromium/1810
loop3    7:3    0  55.5M  1 loop /snap/core18/2253
loop4    7:4    0 217.4M  1 loop /snap/code/81
sda      8:0    0     8G  0 disk 
sdb      8:16   0    30G  0 disk 
└─sdb1   8:17   0    30G  0 part /
sr0     11:0    1  1024M  0 rom  '''

dd if=openwrt-21.02.1-x86-64-generic-ext4-combined.img of=/dev/sda

# 9. close ubuntu, unmont the hard disk file, and mount it back to openwrt vm
# !!! this imange now can run on the vm

#10. check installed packages 

opkg list-installed 
opkg list-installed | grep "usb"
opkg list-installed | grep "kmod"

#11. check disk space. 
df -h

/dev/root 102.4M
/dev/sda1/15.7M






#-#-#-#-#-#-#-#-#-#-#-#-#- install Luci #-#-#-#-#-#-#-#-#-
opkg update && opkg install luci
#-#-#-#-#-#-#-#-#-#-#-#-#-end of install Luci #-#-#-#-#-#-#-#-#-


#在家里装完软路由以后发现没法remote desktop 到公司的主机。（主机就在家里）
#折腾了半天， 一开始以为是OpenWRT的设置问题。后来又用了不同的路由测试都不行。
#最后发现是公司电脑的防火墙设置改变了， 在控制面版--防火墙---允许的应用--找到remote desktop --- 选择public/private
#另外我还在allow remote desktop 的设置里，允许访问的人里加入了Everyone。（不知道这个设置是否有用）

#2021.12.16 update
#今天无法访问网络，检查了一下， 发现openwrt的 /overlay 空间为0， AdGuard Home 停止工作。
#比较笨的解决办法是，今入openwrt， 系统 ==》备份/升级 ==》 恢复出厂设置。
#恢复后 ，修改openwrt的IP地址，然后重新配置USB Wan共享。这次并没有再开启AdGuardHome