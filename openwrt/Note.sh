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


block info （显示固件信息）

/dev/loop0: UUID="5de7e234-6913-43b6-8e3a-d26f587f76d5" LABEL="rootfs_data" VERSION="1.14" MOUNT="/overlay" TYPE="f2fs"
/dev/sda1: UUID="84173db5-fa99-e35a-95c6-28613cc79ea9" LABEL="kernel" VERSION="1.0" MOUNT="/boot" TYPE="ext4"
/dev/sda2: UUID="488a811c-6314f3f2-5e697022-95d0cdbb" VERSION="4.0" MOUNT="/rom" TYPE="squashfs"
/dev/sda3: UUID="3cf63333-2e59-432b-8839-ac7cf5b825d1" VERSION="1.0" TYPE="ext4"


df -h

Filesystem                Size      Used Available Use% Mounted on
/dev/root                 4.0M      4.0M         0 100% /rom
tmpfs                     2.9G    232.0K      2.9G   0% /tmp
/dev/loop0               98.1M     96.7M      1.3M  99% /overlay
overlayfs:/overlay       98.1M     96.7M      1.3M  99% /
/dev/sda1                15.7M      4.8M     10.7M  31% /boot
/dev/sda1                15.7M      4.8M     10.7M  31% /boot
tmpfs                   512.0K         0    512.0K   0% /dev



vim /etc/rc.local （开机的自动批处理）

    i键，开始编辑
    Esc 退出编辑
    ：w保存
    ：q退出


    mount /dev/sda3 /overlay
    usbmuxd
    /opt/AdGuardHome/AdGuardHome #如果限制性这个，后面的语句就无法执行了

    exit 0


    uci show network
    uci set fstab.mount xxx = ""
    uci commit fstab
    uci -q delete fstab.overlay

    reboot



#硬盘增加分区，+ 挂载overlay流程

opkg update && opkg install cfdisk

cfdisk

#Free Space  / New / primary / write / yes

block info

mkfs.ext4 /dev/sda3

mount /dev/sda3 /mnt/sda3

cp -r /overlay/* /mnt/sda3

mount /dev/sda3 /overlay

#AdGuardHome install
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
