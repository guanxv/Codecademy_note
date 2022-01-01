#how to crack a linux password. 

go to /etc folder. find passwd and shadow file.

when open shadow. you cound find

    root:$1$60a6wQfe$XQEdtkH8HMvQis1IZIwT50:16384:0:99999:7:::
    guanxv:$6$TYEUnI5fP7zjLSTk$fVyPSA19N1IDoDAAdB8orxfjk3gEr20PxG5tfJ3ZobD7a2Up532LXRo1JuSJ7CV0jS6Obdj4LnV3WLIYU/S0:18656:0:99999:7:::

$1$ is MD5
$2a$ is Blowfish
$2y$ is Blowfish
$5$ is SHA-256
$6$ is SHA-512

if the password is simple enough , you can use

john --fork=4 shadow # to crack the password.

but most of the times, the password is complex, you need to use the hackcat to crack the password.

hackcat is able to use GPU, so the speed is very fast.


sample command:

hackcat -a3 -m500 "$1$60a6wQfe$XQEdtkH8HMvQis1IZIwT50" 

    #$1$ indicate it is MD5. 
    #60a6wQfe$ is the salt, 
    #XQEdtkH8HMvQis1IZIwT50 is the actaul hash of the password.

hackcat is able to crack many types of password. see help for more inforamtion 

sample of using mask attack

hashcat -a 3 -m 0 example0.hash ?a?a?a?a?a?a #will try all the 6 keyspaces (uper lower letter + digit + special)


  ? | Charset
 ===+=========
  l | abcdefghijklmnopqrstuvwxyz
  u | ABCDEFGHIJKLMNOPQRSTUVWXYZ
  d | 0123456789
  h | 0123456789abcdef
  H | 0123456789ABCDEF
  s |  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  a | ?l?u?d?s
  b | 0x00 - 0xff


# by the way to extrac a firmware and modify and packing back to a bin file see this link https://www.youtube.com/watch?v=hV8W4o-Mu2o&t=314s

#sample command using mask to crack shadow
hashcat -a3 -m500 -o result.txt shadow ./Corporate_Masks-master/corp_8.hcmask
'''
Session..........: hashcat
Status...........: Running
Hash.Type........: md5crypt, MD5 (Unix), Cisco-IOS $1$ (MD5)
Hash.Target......: $1$60a6wQfe$XQEdtkH8HMvQis1IZIwT50
Time.Started.....: Sun Jan  2 09:54:51 2022 (40 secs)
Time.Estimated...: Sun Jan  2 09:58:05 2022 (2 mins, 34 secs)
Guess.Mask.......: ?l?u?d?d?d?d?d?d [8]
Guess.Queue......: 41/21990 (0.19%)
Speed.#1.........:  3460.6 kH/s (199.42ms) @ Accel:1024 Loops:1000 Thr:32 Vec:1
Recovered........: 0/1 (0.00%) Digests, 0/1 (0.00%) Salts
Progress.........: 139853824/676000000 (20.69%)
Rejected.........: 0/139853824 (0.00%)
Restore.Point....: 5046272/26000000 (19.41%)
Restore.Sub.#1...: Salt:0 Amplifier:12-13 Iteration:0-1000
Candidates.#1....: hS988577 -> hI993107
Hardware.Mon.#1..: Temp: 75c Fan: 80% Util:100% Core:1860MHz Mem:6801MHz Bus:16

'''


hashcat -a0 -m500 -o result.txt shadow rockyou.txt -r ./rules/best64.rule --loopback

hashcat -a0 -m500 -o result.txt shadow realuniq.lst -r ./rules/best64.rule --loopback



# speed, with GTX 1660 , can crack 8 dig, digtial in 20 sec.
